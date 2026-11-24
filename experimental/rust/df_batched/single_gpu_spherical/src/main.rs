//! `df-batched-spherical`: build the density-fitting factor B of a molecule
//! on one GPU, batched over shell pairs and fitted on the card, over
//! FelizJueves's fully spherical `ssperi3` kernels with no spherical
//! transform (`ssperi3.rs`), and report what it took. `--baseline` builds
//! it with rust/'s `add_packed_b` instead (`speri3` and the transform), in
//! the same binary, for an A/B.
//!
//! Each run plans, in a device room the caller picks (so the planner's
//! adaptive batch width with it):
//!
//! - **prefix**: the BLAS workspace, the orbital records, the metric (P|Q)
//!   and its Cholesky factor L;
//! - **full**: the prefix, then the build: per class of shell pairs, a
//!   counted loop of batches, each the packed three-center integrals over
//!   every aux function (already spherical on all three indices), the trsm
//!   against L, a transpose and a pitched download into B on the host.
//!
//! B's build time is the full run's wall less the prefix run's, both warm
//! (the first run of each is discarded). Printed as one JSON object on
//! stdout; the plan text goes to `--plan-out` when given.
//!
//! ```text
//! df-batched-spherical --xyz gly10.xyz [--basis cc-pvdz] [--aux cc-pvdz-jkfit]
//!            [--schwarz 1e-12] [--room-mib N]... [--reps 3] [--compare]
//!            [--baseline] [--check]
//!            [--plan-out plan.txt] [--record-out timing.json]
//! ```
//!
//! Several `--room-mib` run one after another in one process, the width
//! sweep; with `--compare` each run's B is held against the first's. With
//! `--check`, before any timing, B is built once each way at the first
//! room and the spherical one held against rust/'s, element by element.

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
use fb_integrals_eri_density_fitting_three_center::add_packed_b;
use fb_integrals_mmd::record_tables;
use fb_recipes_cholesky::add_cholesky;
use serde_json::{Value, json};

/// The command line.
struct Args {
    xyz: String,
    basis: String,
    aux: String,
    schwarz: f64,
    rooms: Vec<usize>,
    reps: usize,
    compare: bool,
    baseline: bool,
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
        rooms: Vec::new(),
        reps: 3,
        compare: false,
        baseline: false,
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
            "--room-mib" => a.rooms.push(value().parse::<usize>().expect("MiB") << 20),
            "--reps" => a.reps = value().parse().expect("a count"),
            "--compare" => a.compare = true,
            "--baseline" => a.baseline = true,
            "--check" => a.check = true,
            "--plan-out" => a.plan_out = Some(value()),
            "--record-out" => a.record_out = Some(value()),
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(!a.xyz.is_empty(), "--xyz is required");
    assert!(a.reps > 0, "--reps is at least 1");
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

/// Plan the prefix, or with `build` the prefix and B, in `room` device
/// bytes (the BLAS workspace on top); with `keep`, B copied out at the end.
/// B is built by `ssperi3::add_packed_b_spherical`, or with `baseline` by
/// rust/'s `add_packed_b`.
fn plan(
    basis: &BasisSet,
    aux: &BasisSet,
    map: &PairMap,
    room: usize,
    build: bool,
    keep: bool,
    baseline: bool,
) -> (Plan<'static>, Option<Arc<Mutex<Vec<f64>>>>) {
    let naux = aux.nao_sphe() as usize;
    let room = Room {
        device: room + BLAS_WORKSPACE_BYTES,
        pinned: usize::MAX,
    };
    Dag::plan_within(
        |_| Ok(room),
        |dag| {
            dag.blas_workspace(0);
            let classes = fb_integrals_mmd::add_records(dag, 0, basis)?;
            let (mut u, mut u_aux) = (None, None);
            let metric = add_metric(dag, 0, aux, &mut u_aux)?;
            let factor = add_cholesky(dag, 0, "L", naux, metric)?;
            if !build {
                return Ok(None);
            }
            let b = if baseline {
                add_packed_b(
                    dag, 0, basis, &classes, aux, map, factor, &mut u, &mut u_aux,
                )?
            } else {
                ssperi3::add_packed_b_spherical(dag, 0, basis, &classes, aux, map, factor)?
            };
            if !keep {
                return Ok(None);
            }
            let out = Arc::new(Mutex::new(Vec::new()));
            dag.add_cpu("keep B", 0, Keep(b, Arc::clone(&out)));
            Ok(Some(out))
        },
    )
    .expect("the plan is made")
}

/// Per task name: count, host and device time summed, over `record`.
fn by_task(record: &Record) -> Value {
    let mut out = serde_json::Map::new();
    for t in &record.tasks {
        // A class's batch tasks share a name but for "(la, lb)": keep it,
        // the per-class breakdown is the point.
        let e = out
            .entry(t.name.clone())
            .or_insert_with(|| json!({"count": 0, "host_s": 0.0, "device_s": 0.0}));
        e["count"] = json!(e["count"].as_u64().unwrap() + 1);
        e["host_s"] = json!(e["host_s"].as_f64().unwrap() + t.host.as_secs_f64());
        let d = t.device.map_or(0.0, |d| d.as_secs_f64());
        e["device_s"] = json!(e["device_s"].as_f64().unwrap() + d);
    }
    Value::Object(out)
}

/// The plan's width line and each counted loop's trips.
fn plan_shape(text: &str) -> (Option<String>, Vec<usize>) {
    let width = text
        .lines()
        .find(|l| l.starts_with("width packed B batch"))
        .map(|l| l.trim().to_owned());
    let trips = text
        .lines()
        .filter_map(|l| {
            let at = l.find("closes counted loop")?;
            let rest = &l[at..];
            let n = rest.split(", ").nth(1)?.split(' ').next()?;
            n.parse().ok()
        })
        .collect();
    (width, trips)
}

/// B built once by `plan` with `keep`, run once.
fn build_once(
    basis: &BasisSet,
    aux: &BasisSet,
    map: &PairMap,
    room: usize,
    baseline: bool,
) -> Vec<f64> {
    let (planned, kept) = plan(basis, aux, map, room, true, true, baseline);
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let r = instance.run(&mut backend, 1);
    r.result.as_ref().expect("the run succeeds");
    let kept = kept.expect("kept");
    std::mem::take(&mut *kept.lock().unwrap())
}

/// The spherical B against rust/'s: the largest absolute difference, where
/// it is (column, Q), the largest |B|, and the largest difference per
/// class `(la, lb)` relative to that class's largest |B|.
fn check(basis: &BasisSet, aux: &BasisSet, map: &PairMap, room: usize, naux: usize) -> Value {
    let base = build_once(basis, aux, map, room, true);
    let sph = build_once(basis, aux, map, room, false);
    assert_eq!(base.len(), sph.len(), "B's length");
    let n_columns = map.sphe().n_columns();
    let (mut max, mut at, mut big) = (0.0_f64, (0, 0), 0.0_f64);
    let mut nonfinite = 0usize;
    for (i, (x, y)) in base.iter().zip(&sph).enumerate() {
        if !y.is_finite() {
            nonfinite += 1;
        }
        let d = (x - y).abs();
        if d > max || d.is_nan() {
            max = d;
            at = (i % n_columns, i / n_columns);
        }
        big = big.max(x.abs());
    }
    let sphe = map.sphe().blocks();
    let mut classes = Vec::new();
    for c in map.classes() {
        let first = sphe[c.pairs.start].column;
        let last = &sphe[c.pairs.end - 1];
        let cols = first..last.column + last.n[0] * last.n[1];
        let (mut d, mut m) = (0.0_f64, 0.0_f64);
        for q in 0..naux {
            for col in cols.clone() {
                let i = col + n_columns * q;
                d = d.max((base[i] - sph[i]).abs());
                m = m.max(base[i].abs());
            }
        }
        classes.push(json!({
            "la": c.la, "lb": c.lb, "pairs": c.pairs.len(),
            "max_abs_diff": d, "max_abs": m, "rel": if m > 0.0 { d / m } else { 0.0 },
        }));
    }
    json!({
        "max_abs_diff": max,
        "at_column_q": [at.0, at.1],
        "max_abs": big,
        "rel": max / big,
        "nonfinite": nonfinite,
        "bitwise": base.iter().zip(&sph).all(|(x, y)| x.to_bits() == y.to_bits()),
        "classes": classes,
    })
}

/// Run `instance` `reps + 1` times, the first discarded; the walls, and the
/// last run's record.
fn timed(planned: Plan<'static>, reps: usize) -> (Vec<Duration>, Record) {
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let warm = instance.run(&mut backend, 1);
    warm.result.as_ref().expect("the warm-up run succeeds");
    let mut walls = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let r = instance.run(&mut backend, 1);
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

fn main() {
    let a = args();
    let (basis, aux) = bases(&a);
    let map = pair_map(&basis, a.schwarz);
    let naux = aux.nao_sphe() as usize;
    let columns = map.sphe().n_columns();
    let (free, total) = fb_core_arena::device_memory(0).expect("the card reads");
    let rooms = if a.rooms.is_empty() {
        // The card's free memory less a margin and the workspace.
        vec![free.saturating_sub((512 << 20) + BLAS_WORKSPACE_BYTES)]
    } else {
        a.rooms.clone()
    };
    eprintln!(
        "nao {}, naux {naux}, kept pairs {}, columns {columns}, B {:.3} GB; card {:.2} of {:.2} GB free",
        basis.nao_sphe(),
        map.pairs().len(),
        8.0 * (naux * columns) as f64 / 1e9,
        free as f64 / 1e9,
        total as f64 / 1e9
    );

    let checked = if a.check {
        let c = check(&basis, &aux, &map, rooms[0], naux);
        eprintln!(
            "check against rust/'s add_packed_b: max |dB| {:.3e} of max |B| {:.3e} (rel {:.3e}), \
             {} non-finite",
            c["max_abs_diff"].as_f64().unwrap(),
            c["max_abs"].as_f64().unwrap(),
            c["rel"].as_f64().unwrap(),
            c["nonfinite"]
        );
        c
    } else {
        Value::Null
    };

    let (prefix_plan, _) = plan(&basis, &aux, &map, rooms[0], false, false, a.baseline);
    let prefix_device = prefix_plan.device_table(0).total();
    let (prefix_walls, prefix_record) = timed(prefix_plan, a.reps);
    let prefix = median(&prefix_walls);
    eprintln!("prefix (records, metric, Cholesky): {prefix:.3} s");

    let mut first_b: Option<Vec<f64>> = None;
    let mut runs = Vec::new();
    for (i, &room) in rooms.iter().enumerate() {
        let (planned, kept) = plan(&basis, &aux, &map, room, true, a.compare, a.baseline);
        let text = planned.to_string();
        let (width, trips) = plan_shape(&text);
        if let Some(path) = &a.plan_out {
            let path = if rooms.len() > 1 {
                format!("{path}.{i}")
            } else {
                path.clone()
            };
            std::fs::write(path, &text).expect("the plan text writes");
        }
        let device = planned.device_table(0).total();
        let pinned = planned.host_table(0).total();
        let (walls, record) = timed(planned, a.reps);
        let wall = median(&walls);
        let keep_s: f64 = record
            .tasks
            .iter()
            .filter(|t| t.name == "keep B")
            .map(|t| t.host.as_secs_f64())
            .sum();
        let build = wall - keep_s - prefix;
        let mut compare = Value::Null;
        if let Some(kept) = kept {
            let b = std::mem::take(&mut *kept.lock().unwrap());
            match &first_b {
                None => first_b = Some(b),
                Some(f) => {
                    let (mut max, mut big) = (0.0_f64, 0.0_f64);
                    for (x, y) in f.iter().zip(&b) {
                        max = max.max((x - y).abs());
                        big = big.max(x.abs());
                    }
                    let bitwise = f.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
                    compare = json!({"max_abs_diff": max, "max_abs": big, "bitwise": bitwise});
                }
            }
        }
        eprintln!(
            "room {:.0} MiB: {} | {} loops, {} trips | device {:.2} GB | build {build:.3} s",
            room as f64 / (1 << 20) as f64,
            width.as_deref().unwrap_or("no width line"),
            trips.len(),
            trips.iter().sum::<usize>(),
            device as f64 / 1e9
        );
        if let Some(path) = &a.record_out {
            let path = if rooms.len() > 1 {
                format!("{path}.{i}")
            } else {
                path.clone()
            };
            std::fs::write(path, record.to_json()).expect("the record writes");
        }
        runs.push(json!({
            "room_bytes": room,
            "width_line": width,
            "trips": trips,
            "device_bytes": device,
            "pinned_bytes": pinned,
            "walls_s": secs(&walls),
            "wall_s": wall,
            "keep_s": keep_s,
            "build_s": build,
            "tasks": by_task(&record),
            "compare": compare,
        }));
    }

    let out = json!({
        "xyz": a.xyz,
        "basis": a.basis,
        "aux": a.aux,
        "schwarz": a.schwarz,
        "path": if a.baseline { "baseline (speri3 + transform)" } else { "spherical (ssperi3)" },
        "check": checked,
        "nao": basis.nao_sphe(),
        "naux": naux,
        "kept_pairs": map.pairs().len(),
        "classes": map.classes().len(),
        "columns": columns,
        "b_bytes": 8 * naux * columns,
        "card_total_bytes": total,
        "card_free_bytes": free,
        "prefix": {
            "device_bytes": prefix_device,
            "walls_s": secs(&prefix_walls),
            "wall_s": prefix,
            "tasks": by_task(&prefix_record),
        },
        "runs": runs,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
