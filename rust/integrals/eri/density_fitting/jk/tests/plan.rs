//! J's and K's plans at the fixtures' shape, naux = 23, nao = 7, nocc = 3
//! (tier 3): `df coulomb` after B's and D's producers, γ its scratch of
//! `naux` doubles and J an output of `nao²`; `df exchange` after B's and
//! C's, X its scratch of `naux · nocc · nao` doubles and K an output of
//! `nao²`. Each plan text, the worker's device table among it, is a
//! committed golden under `tests/golden/`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). The streamed pass's plan, one J and
//! one K over a packed B of naux = 150 in a heap row, is a golden too, and
//! narrowed below a batch of 32 it logs a `warn`. Planning touches no
//! device; K's does load `libfb_recipes_kernels.so`, and the streamed
//! pass `libfb_integrals_kernels.so` too, whose load checks it runs.

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask, Plan, Room};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_jk::{
    Contraction, add_df_coulomb, add_df_exchange, add_streamed_jk,
};
use fb_integrals_eri_density_fitting_pair_map::{Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::add_packed_tables;
use fb_test_support::golden;

/// The fitting set's size.
const NAUX: usize = 23;
/// The orbital basis's size.
const NAO: usize = 7;
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

/// B's producer and one density's (`name`, `len` doubles), on worker 0.
fn inputs(dag: &mut Dag<'_>, name: &'static str, len: usize) -> [RowId<Device, f64>; 2] {
    [("B", NAUX * NAO * NAO), (name, len)]
        .map(|(name, len)| dag.add_gpu(name, 0, Input { name, len }))
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

#[test]
fn coulomb_plan_golden() {
    let mut dag = Dag::new();
    let [b, d] = inputs(&mut dag, "D", NAO * NAO);
    let j = add_df_coulomb(&mut dag, 0, NAO, NAUX, b, d);
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(j), NAO * NAO);
    assert_eq!(planned.predecessors(2), [0, 1]);
    plan_golden(&planned, "plan_coulomb_naux23_nao7.txt");
}

#[test]
fn exchange_plan_golden() {
    let mut dag = Dag::new();
    let [b, c] = inputs(&mut dag, "C", NAO * NOCC);
    let k = add_df_exchange(&mut dag, 0, NAO, NAUX, NOCC, b, c).unwrap();
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(k), NAO * NAO);
    assert_eq!(planned.predecessors(2), [0, 1]);
    plan_golden(&planned, "plan_exchange_naux23_nao7_nocc3.txt");
}

/// A factor of no columns is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a factor of at least one column")]
fn an_empty_factor_is_refused() {
    let mut dag = Dag::new();
    let [b, c] = inputs(&mut dag, "C", NAO);
    let _ = add_df_exchange(&mut dag, 0, NAO, NAUX, 0, b, c);
}

/// The streamed pass's fitting set: two full batches of 64 and a short
/// one of 22 at the widest batch.
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

/// One J and one K streamed over the packing's B on worker 0, planned in
/// `room` (none: [`Dag::plan`], the widest batch), and the one width line
/// of its text.
fn streamed(room: Option<Room>) -> (Plan<'static>, String) {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    let declare = |dag: &mut Dag<'static>| {
        let b = dag.add_cpu("source", 0, HeapB(NAUX_STREAMED * nc));
        let p = packing.clone();
        let tables = add_packed_tables(dag, 0, nao, nc, move |r, c, w| p.write_tables(r, c, w));
        let [d, c] = [("D", nao * nao), ("C", nao * NOCC)]
            .map(|(name, len)| dag.add_gpu(name, 0, Input { name, len }));
        let set = [
            Contraction::Coulomb(d),
            Contraction::Exchange { c, nocc: NOCC },
        ];
        add_streamed_jk(dag, 0, &packing, tables, NAUX_STREAMED, b, &set).map(drop)
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

/// The pass at the widest batch: `df jk pack`, the ring of two full
/// batches (stage, upload on lane 1, `df jk batch`), the short batch's
/// loop of one trip, and `df jk finish`; the region two copies of a
/// batch's rows, and no device row of B whole, `naux · nao²` doubles, or
/// of its packed `naux · n_columns`.
#[test]
fn streamed_plan_golden() {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    let (planned, width) = streamed(None);
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
    let widest = streamed(None).0;
    let mut device = widest.device_table(0).total();
    let narrowed = |room: Room| {
        let before = fb_core_log::warning_count();
        let (_, width) = streamed(Some(room));
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
