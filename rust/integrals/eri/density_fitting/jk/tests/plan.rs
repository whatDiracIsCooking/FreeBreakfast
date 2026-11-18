//! The streamed pass's plan (tier 3), one J and one K over a packed B of
//! naux = 150 in a heap row: its text, the worker's device table among it,
//! is the committed golden `tests/golden/plan_streamed_naux150_nao9_nocc3.txt`,
//! `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"); narrowed
//! below a batch of 32 it logs a `warn`; a fitting set of one batch is one
//! trip; a factor of no columns is refused. Planning touches no device; it
//! does load `libfb_recipes_kernels.so` and `libfb_integrals_kernels.so`,
//! whose load checks it runs.

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask, Plan, Room};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_jk::{Contraction, add_streamed_jk};
use fb_integrals_eri_density_fitting_pair_map::{Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::add_packed_tables;
use fb_test_support::golden;

/// C's columns.
const NOCC: usize = 3;

/// GPU: an input, declared and never computed.
struct Input {
    name: &'static str,
    len: usize,
}

impl GpuTask for Input {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

/// `planned`'s text, each line's trailing whitespace stripped (an empty
/// handoffs line ends in a space, which the trailing-whitespace hook strips
/// from the committed file), compared with the golden `file`.
fn plan_golden(planned: &Plan<'_>, file: &str) {
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), file, &text);
}

/// The streamed pass's fitting set: two full batches of 64 and a ragged
/// last one of 22 at the widest batch.
const NAUX_STREAMED: usize = 150;

/// CPU: the packed B's heap row, declared and never filled.
struct HeapB(usize);

impl CpuTask for HeapB {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("B", self.0)
    }

    fn run(&self, _: &Self::Rows, _: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        unreachable!("the plan is never run")
    }
}

/// The spherical packing of an s, a p and a d shell on two centres, every
/// pair kept: 9 functions, 45 columns.
fn packing() -> Packing {
    let at = |z| Double3 { x: 0.0, y: 0.0, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(0, &[1.0], &[1.0], at(0.0)),
        Shell::zipped(1, &[0.7], &[1.0], at(0.0)),
        Shell::zipped(2, &[0.5], &[1.0], at(1.5)),
    ]);
    PairMap::new(&record_tables(&basis), None, 0.0)
        .sphe()
        .clone()
}

/// One J and one K streamed over the packing's B of `naux` aux functions
/// on worker 0, planned in `room` (none: [`Dag::plan`], the widest
/// batch), and the one width line of its text.
fn streamed(naux: usize, room: Option<Room>) -> (Plan<'static>, String) {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    let declare = |dag: &mut Dag<'static>| {
        let b = dag.add_cpu("source", 0, HeapB(naux * nc));
        let p = packing.clone();
        let tables = add_packed_tables(dag, 0, nao, nc, move |r, c, w| p.write_tables(r, c, w));
        let [d, c] = [("D", nao * nao), ("C", nao * NOCC)]
            .map(|(name, len)| dag.add_gpu(name, 0, Input { name, len }));
        let set = [
            Contraction::Coulomb(d),
            Contraction::Exchange { c, nocc: NOCC },
        ];
        add_streamed_jk(dag, 0, &packing, tables, naux, b, &set).map(drop)
    };
    let planned = match room {
        None => {
            let mut dag = Dag::new();
            declare(&mut dag).unwrap();
            dag.plan()
        }
        Some(room) => Dag::plan_within(|_| Ok(room), declare).unwrap().0,
    };
    let width = planned
        .to_string()
        .lines()
        .find(|l| l.starts_with("width df jk batch: "))
        .expect("the plan has the batch's width line")
        .to_owned();
    (planned, width)
}

/// The pass at the widest batch: `df jk pack`, the ring of three
/// batches, the last ragged (stage, upload on lane 1, `df jk batch`), and
/// `df jk finish`; the region two copies of a
/// batch's rows, and no device row of B whole, `naux · nao²` doubles, or
/// of its packed `naux · n_columns`.
#[test]
fn streamed_plan_golden() {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    let (planned, width) = streamed(NAUX_STREAMED, None);
    assert!(width.contains(": 64 of 64,"), "{width}");
    for whole in [NAUX_STREAMED * nao * nao, NAUX_STREAMED * nc] {
        let bytes = whole * size_of::<f64>();
        let rows = planned.device_table(0).rows();
        assert!(
            rows.iter().all(|r| r.bytes() < bytes),
            "a device row of {whole} doubles or more"
        );
    }
    plan_golden(&planned, "plan_streamed_naux150_nao9_nocc3.txt");
}

/// A fitting set no larger than one batch is one trip, its ring one copy:
/// the pass plans, a loop of one trip at depth 1. Catches a ring deeper
/// than the loop's trips, which the planner refuses (an H atom's DF-ROHF).
#[test]
fn a_fitting_set_of_one_batch_is_one_trip() {
    let (planned, width) = streamed(20, None);
    assert!(width.contains(": 64 of 64,"), "{width}");
    let text = planned.to_string();
    assert!(
        text.lines()
            .any(|l| l.contains("df jk batch:") && l.contains("1 trips, depth 1")),
        "{text}"
    );
}

/// The batch's width in a plan's width line.
fn width_of(line: &str) -> usize {
    line["width df jk batch: ".len()..]
        .split(' ')
        .next()
        .and_then(|w| w.parse().ok())
        .unwrap()
}

/// Narrowed by its device room below a batch of 32, the pass's plan logs
/// one `warn` naming the width, and none at 32 or more: the room from the
/// widest batch's plan's total down, an eighth at a time. Then the pinned
/// room narrows it too: a quarter of the widest plan's pinned table, its
/// windows' ring nearly all of it, leaves at most 16, and warns. One case,
/// since the warnings are counted per process.
#[test]
fn a_narrowed_streamed_plan_warns() {
    fb_core_log::init();
    let widest = streamed(NAUX_STREAMED, None).0;
    let mut device = widest.device_table(0).total();
    let narrowed = |room: Room| {
        let before = fb_core_log::warning_count();
        let (_, width) = streamed(NAUX_STREAMED, Some(room));
        (width, fb_core_log::warning_count() - before)
    };
    loop {
        let (width, warned) = narrowed(Room {
            device,
            pinned: usize::MAX,
        });
        if width_of(&width) < 32 {
            assert_eq!(warned, 1, "{width}");
            break;
        }
        assert_eq!(warned, 0, "{width}");
        device -= device / 8;
    }
    let (width, warned) = narrowed(Room {
        device: usize::MAX,
        pinned: widest.host_table(0).total() / 4,
    });
    assert!((1..=16).contains(&width_of(&width)), "{width}");
    assert_eq!(warned, 1, "{width}");
}

/// A factor of no columns is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "every factor has a column")]
fn an_empty_factor_is_refused() {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    let mut dag = Dag::new();
    let b = dag.add_cpu("source", 0, HeapB(NAUX_STREAMED * nc));
    let p = packing.clone();
    let tables = add_packed_tables(&mut dag, 0, nao, nc, move |r, c, w| p.write_tables(r, c, w));
    let c = dag.add_gpu(
        "C",
        0,
        Input {
            name: "C",
            len: nao,
        },
    );
    let set = [Contraction::Exchange { c, nocc: 0 }];
    let _ = add_streamed_jk(&mut dag, 0, &packing, tables, NAUX_STREAMED, b, &set);
}
