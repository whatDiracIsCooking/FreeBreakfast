//! `df-batched-multi-gpu`: build the density-fitting factor B of a molecule
//! over several GPUs, each card a share of every class's shell pairs
//! (`ssperi3.rs`), and report what it took against one card.
//!
//! Each run plans, on `n` cards, in a device room per card the caller
//! picks:
//!
//! - **prefix**, on every card: the BLAS workspace, the orbital records,
//!   the metric (P|Q) and its Cholesky factor L;
//! - **full**: the prefix, then each card's part of the build, the counted
//!   loops of `../single_gpu_spherical` over its share of the pairs, into
//!   a heap row of its own.
//!
//! B's build time is the full run's wall less the prefix run's, both warm
//! (the first run of each is discarded). Printed as one JSON object on
//! stdout; the plan text goes to `--plan-out` when given.
//!
//! ```text
//! df-batched-multi-gpu --xyz gly20.xyz [--basis cc-pvdz] [--aux cc-pvdz-jkfit]
//!            [--schwarz 1e-12] [--cards N]... [--room-mib N] [--cpu N]
//!            [--reps 3] [--check]
//!            [--plan-out plan.txt] [--record-out timing.json]
//! ```
//!
//! Several `--cards` run one after another in one process, each on devices
//! `0..N`; with none, 1 and then every card the runtime lists. `--cpu` is
//! the CPU pool's threads (the staging and the unstaging of every card),
//! by default four per card. With `--check`, each run's B, its parts put
//! back in the map's columns, is held against the first run's, element by
//! element; one card's is `../single_gpu_spherical`'s build.

mod ssperi3;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{BLAS_WORKSPACE_BYTES, CpuTask, Dag, Plan, Room};
use fb_core_thread_pool::Record;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_mmd::record_tables;
use fb_recipes_cholesky::add_cholesky;
use serde_json::{Value, json};

/// The command line.
struct Args {
    xyz: String,
    basis: String,
    aux: String,
    schwarz: f64,
    cards: Vec<usize>,
    room: Option<usize>,
    cpu: Option<usize>,
    reps: usize,
    check: bool,
    plan_out: Option<String>,
    record_out: Option<String>,
}

fn args() -> Args {
    let mut a = Args {
        xyz: String::new(),
        basis: "cc-pvdz".into(),
        aux: "cc-pvdz-jkfit".into(),
        schwarz: 1e-12,
        cards: Vec::new(),
        room: None,
        cpu: None,
        reps: 3,
        check: false,
        plan_out: None,
        record_out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} takes a value"));
        match flag.as_str() {
            "--xyz" => a.xyz = value(),
            "--basis" => a.basis = value(),
            "--aux" => a.aux = value(),
            "--schwarz" => a.schwarz = value().parse().expect("a number"),
            "--cards" => a.cards.push(value().parse().expect("a count")),
            "--room-mib" => a.room = Some(value().parse::<usize>().expect("MiB") << 20),
            "--cpu" => a.cpu = Some(value().parse().expect("a count")),
            "--reps" => a.reps = value().parse().expect("a count"),
            "--check" => a.check = true,
            "--plan-out" => a.plan_out = Some(value()),
            "--record-out" => a.record_out = Some(value()),
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(!a.xyz.is_empty(), "--xyz is required");
    assert!(a.reps > 0, "--reps is at least 1");
    assert!(a.cards.iter().all(|&n| n > 0), "--cards is at least 1");
    a
}

/// The orbital and fitting bases over the molecule.
fn bases(a: &Args) -> (BasisSet, BasisSet) {
    let molecule = fb_io_xyz::read_xyz(a.xyz.as_ref(), 0, 1).expect("the XYZ file reads");
    let place = |stamp: Result<fb_basis::BasisStamp>| {
        stamp
            .expect("a carried set")
            .primitive_normalized()
            .place(&molecule)
            .expect("the set covers the molecule")
    };
    (
        place(fb_io_basis::resolve_basis(&a.basis)),
        place(fb_io_basis::resolve_aux_basis(&a.aux)),
    )
}

/// The packed pair map, screened as rust/'s `add_df_eri` screens it: each
/// shell pair's largest Cartesian `sqrt((mn|mn))` against `threshold`.
fn pair_map(basis: &BasisSet, threshold: f64) -> PairMap {
    let tables = record_tables(basis);
    if threshold == 0.0 {
        return PairMap::new(&tables, None, 0.0);
    }
    let pairs: Vec<(u32, u32)> = tables
        .classes()
        .iter()
        .flat_map(|c| c.bra.iter().copied().zip(c.ket.iter().copied()))
        .collect();
    let diagonal = fb_integrals_schwarz::schwarz_diagonal(basis, &pairs);
    let shells = basis.shells();
    let mut at = 0;
    let bounds: Vec<f64> = pairs
        .iter()
        .map(|&(a, b)| {
            let n = (shells[a as usize].function_count_cart()
                * shells[b as usize].function_count_cart()) as usize;
            let block = &diagonal[at..at + n];
            at += n;
            block.iter().copied().fold(0.0, f64::max)
        })
        .collect();
    PairMap::new(&tables, Some(&bounds), threshold)
}

/// CPU: B's heap row copied out after the last batch's write.
struct Keep(RowId<Heap, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Keep {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.0);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap_or_else(|e| e.into_inner()) = heap.get(self.0).to_vec();
        Ok(())
    }
}

/// One card's part as planned: its kept copy, its columns and its spans.
struct Kept {
    b: Arc<Mutex<Vec<f64>>>,
    n_columns: usize,
    spans: Vec<(usize, usize, usize)>,
}

/// Plan the prefix on `rooms.len()` cards, or with `build` the prefix and
/// each card's part of B, card `w` in `rooms[w]` device bytes (the BLAS
/// workspace on top); with `keep`, each part copied out at the end.
fn plan(
    basis: &BasisSet,
    aux: &BasisSet,
    map: &PairMap,
    rooms: &[usize],
    build: bool,
    keep: bool,
) -> (Plan<'static>, Vec<Kept>) {
    let naux = aux.nao_sphe() as usize;
    let n = rooms.len();
    Dag::plan_within(
        |w| {
            Ok(Room {
                device: rooms[w] + BLAS_WORKSPACE_BYTES,
                pinned: usize::MAX,
            })
        },
        |dag| {
            let mut kept = Vec::new();
            // Every card's prefix first, then every card's loops: a card's
            // tasks depend on its own alone, so the plan order between
            // cards orders nothing.
            let mut prefix = Vec::with_capacity(n);
            for w in 0..n {
                dag.blas_workspace(w);
                let classes = fb_integrals_mmd::add_records(dag, w, basis)?;
                let mut u_aux = None;
                let metric = add_metric(dag, w, aux, &mut u_aux)?;
                let factor = add_cholesky(dag, w, "L", naux, metric)?;
                prefix.push((classes, factor));
            }
            if !build {
                return Ok(kept);
            }
            let parts = ssperi3::add_packed_b_parts(dag, basis, aux, map, &prefix)?;
            // The keeps after every loop, so each part lives to the end,
            // as it would under a reader.
            for (w, part) in parts.into_iter().enumerate() {
                if keep {
                    let out = Arc::new(Mutex::new(Vec::new()));
                    dag.add_cpu(
                        format!("keep B part {w}"),
                        w,
                        Keep(part.b, Arc::clone(&out)),
                    );
                    kept.push(Kept {
                        b: out,
                        n_columns: part.n_columns,
                        spans: part.spans,
                    });
                }
            }
            Ok(kept)
        },
    )
    .expect("the plan is made")
}

/// The parts put back in the map's layout: `(column, Q)` at
/// `column + n_columns · Q`.
fn assemble(kept: &[Kept], naux: usize, n_columns: usize) -> Vec<f64> {
    let mut b = vec![f64::NAN; naux * n_columns];
    for k in kept {
        let part = std::mem::take(&mut *k.b.lock().unwrap());
        assert_eq!(part.len(), naux * k.n_columns, "a part's length");
        for q in 0..naux {
            for &(global, local, len) in &k.spans {
                let (to, from) = (global + n_columns * q, local + k.n_columns * q);
                b[to..to + len].copy_from_slice(&part[from..from + len]);
            }
        }
    }
    b
}

/// Per worker (`gpu<w>` with its lanes together, `cpu` the pool's) and
/// task name: count, host and device time summed, over `record`.
fn by_task(record: &Record) -> Value {
    let mut out = serde_json::Map::new();
    for t in &record.tasks {
        let worker = match t.worker.split_once('.') {
            Some((gpu, _)) => gpu,
            None => "cpu",
        };
        let of = out
            .entry(worker.to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .unwrap();
        let e = of
            .entry(t.name.clone())
            .or_insert_with(|| json!({"count": 0, "host_s": 0.0, "device_s": 0.0}));
        e["count"] = json!(e["count"].as_u64().unwrap() + 1);
        e["host_s"] = json!(e["host_s"].as_f64().unwrap() + t.host.as_secs_f64());
        let d = t.device.map_or(0.0, |d| d.as_secs_f64());
        e["device_s"] = json!(e["device_s"].as_f64().unwrap() + d);
    }
    Value::Object(out)
}

/// Each card's device time summed over its tasks.
fn device_s(record: &Record, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for t in &record.tasks {
        let Some(w) = t.worker.strip_prefix("gpu") else {
            continue;
        };
        let w: usize = w.split('.').next().unwrap().parse().expect("a worker");
        out[w] += t.device.map_or(0.0, |d| d.as_secs_f64());
    }
    out
}

/// The plan's width lines and each counted loop's trips.
fn plan_shape(text: &str) -> (Vec<String>, Vec<usize>) {
    let widths = text
        .lines()
        .filter(|l| l.starts_with("width packed B batch"))
        .map(|l| l.trim().to_owned())
        .collect();
    let trips = text
        .lines()
        .filter_map(|l| {
            let at = l.find("closes counted loop")?;
            let rest = &l[at..];
            let n = rest.split(", ").nth(1)?.split(' ').next()?;
            n.parse().ok()
        })
        .collect();
    (widths, trips)
}

/// Run `planned` on devices `0..n` `reps + 1` times, the first discarded;
/// the walls, and the last run's record.
fn timed(planned: Plan<'static>, n: usize, cpu: usize, reps: usize) -> (Vec<Duration>, Record) {
    let devices: Vec<i32> = (0..n as i32).collect();
    let mut instance = planned.instantiate(&devices);
    let mut backend = instance.backend();
    let warm = instance.run(&mut backend, cpu);
    warm.result.as_ref().expect("the warm-up run succeeds");
    let mut walls = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let r = instance.run(&mut backend, cpu);
        r.result.as_ref().expect("the run succeeds");
        walls.push(r.wall);
        last = Some(r);
    }
    (walls, last.expect("one run"))
}

fn secs(d: &[Duration]) -> Vec<f64> {
    d.iter().map(Duration::as_secs_f64).collect()
}

fn median(d: &[Duration]) -> f64 {
    let mut s = secs(d);
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// The cards the runtime lists: device indices from 0 until one fails to
/// read its memory.
fn listed() -> Vec<(usize, usize)> {
    (0..)
        .map_while(|d| fb_core_arena::device_memory(d).ok())
        .collect()
}

fn main() {
    let a = args();
    let (basis, aux) = bases(&a);
    let map = pair_map(&basis, a.schwarz);
    let naux = aux.nao_sphe() as usize;
    let columns = map.sphe().n_columns();
    let cards = listed();
    assert!(!cards.is_empty(), "no card");
    let counts = if a.cards.is_empty() {
        if cards.len() > 1 {
            vec![1, cards.len()]
        } else {
            vec![1]
        }
    } else {
        a.cards.clone()
    };
    assert!(
        counts.iter().all(|&n| n <= cards.len()),
        "--cards past the {} listed",
        cards.len()
    );
    // Each card's room: the caller's, else its free memory less a margin
    // and the workspace.
    let rooms: Vec<usize> = cards
        .iter()
        .map(|&(free, _)| {
            a.room
                .unwrap_or(free.saturating_sub((512 << 20) + BLAS_WORKSPACE_BYTES))
        })
        .collect();
    eprintln!(
        "nao {}, naux {naux}, kept pairs {}, columns {columns}, B {:.3} GB; {} cards, {:.2} GB \
         free on the first",
        basis.nao_sphe(),
        map.pairs().len(),
        8.0 * (naux * columns) as f64 / 1e9,
        cards.len(),
        cards[0].0 as f64 / 1e9
    );

    let mut first_b: Option<Vec<f64>> = None;
    let mut runs = Vec::new();
    for (i, &n) in counts.iter().enumerate() {
        let rooms = &rooms[..n];
        let cpu = a.cpu.unwrap_or(4 * n);
        let (prefix_plan, _) = plan(&basis, &aux, &map, rooms, false, false);
        let (prefix_walls, _) = timed(prefix_plan, n, cpu, a.reps);
        let prefix = median(&prefix_walls);

        let (planned, kept) = plan(&basis, &aux, &map, rooms, true, a.check);
        let text = planned.to_string();
        let (widths, trips) = plan_shape(&text);
        let suffix = |path: &str| {
            if counts.len() > 1 {
                format!("{path}.{i}")
            } else {
                path.to_owned()
            }
        };
        if let Some(path) = &a.plan_out {
            std::fs::write(suffix(path), &text).expect("the plan text writes");
        }
        let device: Vec<usize> = (0..n).map(|w| planned.device_table(w).total()).collect();
        let pinned: Vec<usize> = (0..n).map(|w| planned.host_table(w).total()).collect();
        let (walls, record) = timed(planned, n, cpu, a.reps);
        let wall = median(&walls);
        // The keeps run on the pool side by side: their longest, not their
        // sum, is what the wall holds of them (an upper bound of it).
        let keep_s = record
            .tasks
            .iter()
            .filter(|t| t.name.starts_with("keep B part"))
            .map(|t| t.host.as_secs_f64())
            .fold(0.0, f64::max);
        let build = wall - keep_s - prefix;
        let mut compare = Value::Null;
        if a.check {
            let b = assemble(&kept, naux, columns);
            let unset = b.iter().filter(|x| x.is_nan()).count();
            match &first_b {
                None => {
                    compare = json!({"reference": true, "unset_or_nan": unset});
                    first_b = Some(b);
                }
                Some(f) => {
                    let (mut max, mut big) = (0.0_f64, 0.0_f64);
                    for (x, y) in f.iter().zip(&b) {
                        max = max.max((x - y).abs());
                        big = big.max(x.abs());
                    }
                    let bitwise = f.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
                    compare = json!({
                        "max_abs_diff": max, "max_abs": big, "rel": max / big,
                        "bitwise": bitwise, "unset_or_nan": unset,
                    });
                    eprintln!(
                        "  B on {n} cards against {}: max |dB| {max:.3e} of max |B| {big:.3e}, \
                         {unset} unset",
                        counts[0]
                    );
                }
            }
        }
        let busy = device_s(&record, n);
        eprintln!(
            "{n} cards: prefix {prefix:.3} s | build {build:.3} s | {} trips | device s per card \
             {busy:.2?}",
            trips.iter().sum::<usize>()
        );
        for w in &widths {
            eprintln!("  {w}");
        }
        if let Some(path) = &a.record_out {
            std::fs::write(suffix(path), record.to_json()).expect("the record writes");
        }
        runs.push(json!({
            "cards": n,
            "cpu_threads": cpu,
            "room_bytes": rooms,
            "width_lines": widths,
            "trips": trips,
            "device_bytes": device,
            "pinned_bytes": pinned,
            "prefix_walls_s": secs(&prefix_walls),
            "prefix_s": prefix,
            "walls_s": secs(&walls),
            "wall_s": wall,
            "keep_s": keep_s,
            "build_s": build,
            "device_s_per_card": busy,
            "tasks": by_task(&record),
            "compare": compare,
        }));
    }

    let out = json!({
        "xyz": a.xyz,
        "basis": a.basis,
        "aux": a.aux,
        "schwarz": a.schwarz,
        "nao": basis.nao_sphe(),
        "naux": naux,
        "kept_pairs": map.pairs().len(),
        "classes": map.classes().len(),
        "columns": columns,
        "b_bytes": 8 * naux * columns,
        "cards_listed": cards.len(),
        "card_total_bytes": cards[0].1,
        "runs": runs,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
