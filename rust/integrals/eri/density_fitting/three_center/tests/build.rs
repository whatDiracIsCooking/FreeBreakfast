//! The build loop, [`add_packed_b`], for water in cc-pVDZ / cc-pVDZ-JKFIT
//! with every pair kept (threshold 0): its plan in a device room that cuts
//! every class but the smallest into batches, a committed golden
//! (`FB_BLESS=1` rewrites it); admission of a planned width, and its
//! refusal of a room short of one pair's batch (tier 3, planning reads the
//! kernel library and touches no device); and the heap B of that plan
//! against its batches planned apart, bitwise, and the one-batch B within
//! a bar (tier 4).

mod common;

use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{BLAS_WORKSPACE_BYTES, CpuTask, Dag, Plan, Room};
use fb_integrals_eri_density_fitting_fit::add_batch_fit;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::{add_packed_b, add_packed_three_center};
use fb_integrals_mmd::record_tables;
use fb_recipes_cholesky::add_cholesky;
use fb_test_support::golden;

/// The device room the golden's plan and the bitwise case are made in,
/// above the metric's peak: a batch of 27 Cartesian columns, so every
/// class but p-d and d-d takes two loops or more, four of them a short last
/// batch.
const ROOM: usize = 750_000;

/// What a planned build hands back: B's heap row, and, when asked, the
/// one-batch reference, each batch's reference with its first column and
/// its columns, and B, all brought to the host after the run.
struct Built {
    b: RowId<Heap, f64>,
    one: Option<Arc<Mutex<Vec<f64>>>>,
    batches: Vec<Batch>,
    heap: Option<Arc<Mutex<Vec<f64>>>>,
}

/// A batch's reference: its first column, its columns, and its B.
struct Batch {
    first: usize,
    columns: usize,
    out: Arc<Mutex<Vec<f64>>>,
}

/// How much [`plan`] adds to the build.
#[derive(Clone, Copy, PartialEq)]
enum Add {
    /// The build alone.
    Build,
    /// The BLAS workspace too, outside the room, as a run needs.
    Workspace,
    /// Then, after the build, the references of a build at this width:
    /// the one-batch B, and each batch of every class, as `add_packed_b`
    /// cuts them, planned outside any loop, the same shapes; and B kept.
    References(usize),
}

/// Plan water's records, metric, Cholesky and build in a device room of
/// `room` bytes (no pinned limit), with what `add` asks.
fn plan(basis: &BasisSet, aux: &BasisSet, room: usize, add: Add) -> (Plan<'static>, Built) {
    let map = PairMap::new(&record_tables(basis), None, 0.0);
    let naux = aux.nao_sphe() as usize;
    let workspace = if add == Add::Build {
        0
    } else {
        BLAS_WORKSPACE_BYTES
    };
    let room = Room {
        device: room + workspace,
        pinned: usize::MAX,
    };
    Dag::plan_within(
        |_| Ok(room),
        |dag| {
            if add != Add::Build {
                dag.blas_workspace(0);
            }
            let classes = fb_integrals_mmd::add_records(dag, 0, basis)?;
            let (mut u, mut u_aux) = (None, None);
            let metric = add_metric(dag, 0, aux, &mut u_aux)?;
            let factor = add_cholesky(dag, 0, "L", naux, metric)?;
            let b = add_packed_b(
                dag, 0, basis, &classes, aux, &map, factor, &mut u, &mut u_aux,
            )?;
            let mut built = Built {
                b,
                one: None,
                batches: Vec::new(),
                heap: None,
            };
            let Add::References(width) = add else {
                return Ok(built);
            };
            let (cart, sphe) = (map.cart().blocks(), map.sphe().blocks());
            let mut fitted = |dag: &mut Dag<'_>, pairs: std::ops::Range<usize>| {
                let first = sphe[pairs.start].column;
                let columns = sphe[pairs.end - 1].end() - first;
                let packed = add_packed_three_center(
                    dag, 0, basis, &classes, aux, &map, pairs, 1, &mut u, &mut u_aux,
                )?;
                let b = add_batch_fit(dag, 0, naux, columns, factor, packed);
                Ok::<_, fb_core_error_handling::Error>(Batch {
                    first,
                    columns,
                    out: common::download(dag, b, naux * columns),
                })
            };
            built.one = Some(fitted(dag, 0..map.pairs().len())?.out);
            for class in map.classes() {
                let block = cart[class.pairs.start].n[0] * cart[class.pairs.start].n[1];
                let per = (width / block).clamp(1, class.pairs.len());
                let mut start = class.pairs.start;
                while start < class.pairs.end {
                    let end = (start + per).min(class.pairs.end);
                    built.batches.push(fitted(dag, start..end)?);
                    start = end;
                }
            }
            let heap = Arc::new(Mutex::new(Vec::new()));
            dag.add_cpu("keep B", 0, Keep(b, Arc::clone(&heap)));
            built.heap = Some(heap);
            Ok(built)
        },
    )
    .expect("water plans")
}

/// CPU: the heap row of B copied out, after the last batch's write.
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

/// The width line of a plan's text.
fn width_line(planned: &Plan<'_>) -> String {
    planned
        .to_string()
        .lines()
        .find(|l| l.starts_with("width packed B batch"))
        .expect("a width line")
        .to_owned()
}

/// Water's build in [`ROOM`]: B is the heap table's row of
/// `naux · n_columns` doubles, no device row is a whole B's or larger, the
/// classes are cut into several counted loops, and the plan text is the
/// committed golden `plan_h2o_packed_b.txt`.
#[test]
fn build_plan_golden() {
    let (basis, aux) = common::water();
    let map = PairMap::new(&record_tables(&basis), None, 0.0);
    let whole = aux.nao_sphe() as usize * map.sphe().n_columns();
    let (planned, built) = plan(&basis, &aux, ROOM, Add::Build);
    assert_eq!(planned.heap_table().len(built.b), whole, "B is on the heap");
    for row in planned.device_table(0).rows() {
        assert!(
            row.bytes() < 8 * whole,
            "{} holds {} bytes, a whole B's {}",
            row.name(),
            row.bytes(),
            8 * whole
        );
    }
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    let loops = text.matches("closes counted loop").count();
    println!("{} in {loops} loops", width_line(&planned));
    assert!(loops > map.classes().len(), "some class takes two loops");
    golden(env!("CARGO_MANIFEST_DIR"), "plan_h2o_packed_b.txt", &text);
}

/// The width planned in a room is admitted in that room: at 1 in a room
/// just above the metric's peak, narrowed in two more, and every class in
/// one batch in a room past them all.
#[test]
fn the_planned_width_is_admitted() {
    let (basis, aux) = common::water();
    for room in [650_000, ROOM, 900_000, 64 << 20] {
        let (planned, _) = plan(&basis, &aux, room, Add::Build);
        let table = planned.device_table(0);
        println!("{}: total {}", width_line(&planned), table.total());
        fb_core_arena::admit(table, room, "gpu0").unwrap_or_else(|e| panic!("room {room}: {e}"));
    }
}

/// A device cap below one pair's batch leaves the width at 1, and the
/// plan is refused by admission as `Allocation`, naming the loops whose
/// region is past the cap on its own.
#[test]
fn a_cap_below_one_pair_is_refused_by_admission() {
    let (basis, aux) = common::water();
    let cap = 64 << 10;
    let (planned, _) = plan(&basis, &aux, cap, Add::Build);
    assert!(
        width_line(&planned).starts_with("width packed B batch: 1 of "),
        "{}",
        width_line(&planned)
    );
    let err = fb_core_arena::admit(planned.device_table(0), cap, "gpu0").unwrap_err();
    assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
    assert!(err.message.contains("loop "), "a loop named in: {err}");
}

/// The bar against the one-batch B, relative to its `max |B|`: #1714's
/// for the packed B against the resident one, by its analysis, a few ulps
/// of the largest element carried through L⁻¹. Not bitwise: the BLAS
/// picks its kernel by shape, the transform's strided-batched count and
/// the fit's column count among it, so a batch of another size than the
/// one-batch build's rounds apart from it (1.06e-14 of max |B| measured
/// on water at this room).
const BAR: f64 = 1e-12;

/// The heap B of the plan in [`ROOM`], built in several batches per class,
/// against the same batches planned apart, outside any loop, bitwise: the
/// same shapes, so the same kernels, and a trip's cells taken from the
/// wrong batch, a pitch or a trip's column offset wrong, the transpose
/// wrong, or an element left unwritten (the heap block's NaN fill)
/// changes bits; and against the one-batch B within [`BAR`], which
/// catches a batch fitted against the wrong columns. Element
/// `(column, Q)` of the heap row against a reference's `(Q, column)`. All
/// in one process, so the BLAS's choice per shape is the same library's.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_heap_b_is_its_batches_bitwise_and_the_one_batch_b_within_the_bar() {
    let (basis, aux) = common::water();
    let map = PairMap::new(&record_tables(&basis), None, 0.0);
    let (naux, n_columns) = (aux.nao_sphe() as usize, map.sphe().n_columns());
    let (sized, _) = plan(&basis, &aux, ROOM, Add::Workspace);
    let line = width_line(&sized);
    let width: usize = line["width packed B batch: ".len()..]
        .split(' ')
        .next()
        .and_then(|w| w.parse().ok())
        .expect("a width");
    let (planned, built) = plan(&basis, &aux, ROOM, Add::References(width));
    assert_eq!(width_line(&planned), line, "the references move no width");
    println!("{line}: {} batches", built.batches.len());
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 2).result.unwrap();
    let take = |out: &Mutex<Vec<f64>>| std::mem::take(&mut *out.lock().unwrap());
    let heap = take(built.heap.as_ref().unwrap());
    let one = take(built.one.as_ref().unwrap());
    assert_eq!(heap.len(), naux * n_columns, "B kept");
    assert_eq!(one.len(), naux * n_columns, "the one-batch B downloaded");

    let mut misses = Vec::new();
    let mut covered = 0;
    for Batch {
        first,
        columns,
        out,
    } in &built.batches
    {
        let batch = take(out);
        assert_eq!(batch.len(), naux * columns, "a batch downloaded");
        covered += columns;
        let differ = (0..*columns)
            .flat_map(|c| (0..naux).map(move |q| (c, q)))
            .filter(|&(c, q)| {
                heap[first + c + n_columns * q].to_bits() != batch[q + naux * c].to_bits()
            })
            .count();
        if differ > 0 {
            misses.push(format!(
                "batch at column {first}, {columns} columns: {differ} elements differ in their bits"
            ));
        }
    }
    assert_eq!(covered, n_columns, "the batches tile B");
    let scale = one.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for c in 0..n_columns {
        for q in 0..naux {
            worst = worst.max((heap[c + n_columns * q] - one[q + naux * c]).abs());
        }
    }
    println!(
        "water: {} batches bitwise their own planned apart; max |Δ| / max |B| against the \
         one-batch B = {:e}, bar {BAR:e}",
        built.batches.len() - misses.len(),
        worst / scale
    );
    if worst > BAR * scale || worst.is_nan() {
        misses.push(format!("max |Δ| = {worst:e} > {:e}", BAR * scale));
    }
    assert!(misses.is_empty(), "water failed:\n{}", misses.join("\n"));
}
