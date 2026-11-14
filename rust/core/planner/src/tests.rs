//! The planner's device-free tests: tasks over the heap kind, run on
//! [`HostOnly`] workers (tier 3).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;

use super::*;
use fb_core_arena::{Bufs, Device, Heap, Layout, RowId};
use fb_core_error_handling::{Error, ErrorKind, Status};
use fb_core_thread_pool::{HostOnly, Lane, Placement};
use fb_test_support::{golden, panic_message};

/// One row declaration, in the order a [`Declares`] makes them.
#[derive(Clone, Copy)]
enum Decl {
    Read(RowId<Device, f64>),
    Out(&'static str, usize),
    Scratch(&'static str, usize),
}

/// A task that only declares device rows, in a given order, and returns
/// the ones it made. For planning tests; its run does nothing.
struct Declares<const GPU: bool>(Vec<Decl>);

impl<const GPU: bool> Task<Heap, ()> for Declares<GPU> {
    type Rows = Vec<RowId<Device, f64>>;
    const GPU: bool = GPU;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let mut made = Vec::new();
        for &decl in &self.0 {
            match decl {
                Decl::Read(r) => {
                    d.read(r);
                }
                Decl::Out(name, len) => made.push(d.output(name, len)),
                Decl::Scratch(name, len) => made.push(d.scratch(name, len)),
            }
        }
        made
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Heap>,
        _: Lane<'_, ()>,
    ) -> Status {
        Ok(())
    }
}

/// `arena::tests::scf_fixture` declared as tasks: the same rows, in the
/// same order, with the solver's `info` flag as the worker's.
fn scf_dag(nbf: usize) -> Dag<'static, Heap, ()> {
    use Decl::*;
    let n2 = nbf * nbf;
    let lwork = 1 + 6 * nbf + 2 * n2;
    let mut dag = Dag::new();
    let _info = dag.device_worker_row::<i32>(0, "solver info", 1);
    let s = dag.add("overlap", 0, Declares::<true>(vec![Out("S", n2)]))[0];
    let h = dag.add("core", 0, Declares::<true>(vec![Out("H", n2)]))[0];
    let x = dag.add(
        "orthogonalizer",
        0,
        Declares::<true>(vec![
            Read(s),
            Out("X", n2),
            Scratch("syevd work (X)", lwork),
        ]),
    )[0];
    let d = dag.add(
        "guess",
        0,
        Declares::<true>(vec![Read(h), Read(x), Out("D", n2)]),
    )[0];
    let f = dag.add(
        "fock",
        0,
        Declares::<true>(vec![Read(h), Read(d), Scratch("J", n2), Out("F", n2)]),
    )[1];
    let c = dag.add(
        "diagonalize",
        0,
        Declares::<true>(vec![
            Read(f),
            Read(x),
            Out("C", n2),
            Scratch("syevd work (F)", lwork),
        ]),
    )[0];
    dag.add(
        "density",
        0,
        Declares::<true>(vec![Read(c), Out("D next", n2)]),
    );
    dag
}

/// The glue declares what the hand-written builder did: the device table
/// is the arena's golden byte for byte, and the whole plan, its tasks'
/// lanes and dependencies with the table, is a golden of its own.
#[test]
fn scf_plan_golden() {
    let planned = scf_dag(24).plan();
    assert_eq!(
        planned.device_table(0).to_string(),
        include_str!("../../arena/tests/golden/arena_scf_fixture.txt")
    );
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "planner_scf.txt",
        &planned.to_string(),
    );
}

/// A node that checks every row it reads against the reference and writes
/// its own output from it: a read that saw bytes not yet written, or
/// already taken over, fails the run naming the row.
struct Node<const GPU: bool> {
    pos: usize,
    /// Each input's row and producer.
    inputs: Vec<(RowId<Heap, u64>, usize)>,
    len: usize,
    scratch: usize,
    expect: Arc<Vec<Vec<u64>>>,
    ran: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
struct NodeRows {
    out: RowId<Heap, u64>,
    scratch: RowId<Heap, u64>,
}

impl<const GPU: bool> Task<Heap, ()> for Node<GPU> {
    type Rows = NodeRows;
    const GPU: bool = GPU;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> NodeRows {
        for &(row, _) in &self.inputs {
            h.read(row);
        }
        NodeRows {
            scratch: h.scratch("scratch", self.scratch),
            out: h.output("out", self.len),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len), ("scratch", self.scratch)]
    }

    fn run(
        &self,
        rows: &NodeRows,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Heap>,
        _: Lane<'_, ()>,
    ) -> Status {
        for &(row, from) in &self.inputs {
            if h.get(row) != self.expect[from].as_slice() {
                return Err(Error::numerical(format!(
                    "node {} read node {from}'s output before it was written or after it was reused",
                    self.pos
                )));
            }
        }
        h.take_mut(rows.scratch).fill(self.pos as u64);
        h.take_mut(rows.out).copy_from_slice(&self.expect[self.pos]);
        self.ran.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A random-enough value of node `pos` at `i`, from its inputs'.
fn mix(pos: usize, i: usize, inputs: &[&[u64]]) -> u64 {
    let mut x = (pos as u64) << 32 | i as u64;
    for input in inputs {
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ input[i % input.len()];
    }
    x
}

/// A DAG of [`Node`]s, each `(worker, gpu, inputs, len, scratch)`, its
/// reference computed serially.
#[derive(Debug)]
struct Spec {
    nodes: Vec<(usize, bool, Vec<usize>, usize, usize)>,
}

impl Spec {
    fn expect(&self) -> Vec<Vec<u64>> {
        let mut out: Vec<Vec<u64>> = Vec::new();
        for (pos, (_, _, inputs, len, _)) in self.nodes.iter().enumerate() {
            let ins: Vec<&[u64]> = inputs.iter().map(|&u| out[u].as_slice()).collect();
            let v = (0..*len).map(|i| mix(pos, i, &ins)).collect();
            out.push(v);
        }
        out
    }

    /// The DAG, and each node's output row.
    fn dag(&self, ran: &Arc<AtomicUsize>) -> (Dag<'static, Heap, ()>, Vec<RowId<Heap, u64>>) {
        let expect = Arc::new(self.expect());
        let mut dag = Dag::new();
        let mut outs: Vec<RowId<Heap, u64>> = Vec::new();
        for (pos, (worker, gpu, inputs, len, scratch)) in self.nodes.iter().enumerate() {
            let inputs = inputs.iter().map(|&u| (outs[u], u)).collect();
            let name = format!("n{pos}");
            macro_rules! node {
                ($gpu:literal) => {
                    Node::<$gpu> {
                        pos,
                        inputs,
                        len: *len,
                        scratch: *scratch,
                        expect: Arc::clone(&expect),
                        ran: Arc::clone(ran),
                    }
                };
            }
            let rows = if *gpu {
                dag.add(name, *worker, node!(true))
            } else {
                dag.add(name, *worker, node!(false))
            };
            outs.push(rows.out);
        }
        (dag, outs)
    }
}

/// A block through the executor, through the glue: tasks on two CPU
/// workers and a stream read and write their rows, a later output takes an
/// earlier one's bytes, ordered only by the handoffs the plan derived, and
/// the plan runs twice over the same blocks with every borrow returned.
/// Each run's record carries every task's dimensions and the blocks'
/// sizes, the plan's own.
#[test]
fn a_dag_through_the_executor() {
    const N: usize = 64;
    let spec = Spec {
        nodes: vec![
            (0, false, vec![], N, 0),
            (0, true, vec![0], N, 0),
            (0, false, vec![0, 1], N, 0),
            (0, true, vec![2], N, 0),
            (0, false, vec![3], N, 0),
        ],
    };
    let ran = Arc::new(AtomicUsize::new(0));
    let (dag, outs) = spec.dag(&ran);
    let planned = dag.plan();
    let table = planned.host_table(0);
    assert_eq!(
        table.offset(outs[3]),
        table.offset(outs[0]),
        "node 3's output reuses node 0's bytes, so a handoff orders them"
    );
    assert_eq!(planned.predecessors(3), &[0, 1, 2]);
    assert_eq!(planned.placement(1), Placement::Stream(0));
    assert_eq!(planned.placement(2), Placement::Cpu);
    let sizes = [
        planned.device_table(0).total(),
        planned.host_table(0).total(),
    ];
    let mut instance = planned.instantiate(&[0]);
    for _ in 0..2 {
        let record = instance.run(&mut HostOnly::new(1), 2);
        record.result.unwrap();
        let arena: Vec<_> = record
            .arena
            .iter()
            .map(|a| (a.worker, a.kind.as_str(), a.bytes))
            .collect();
        assert_eq!(arena, [(0, "device", sizes[0]), (0, "heap", sizes[1])]);
        for t in &record.tasks {
            let dims: Vec<_> = t.dims.iter().map(|(k, &v)| (k.as_str(), v)).collect();
            assert_eq!(dims, [("len", N), ("scratch", 0)], "{}", t.name);
        }
    }
    assert_eq!(ran.load(Ordering::Relaxed), 10);
}

fn arb_spec() -> impl Strategy<Value = Spec> {
    prop::collection::vec(
        (
            0..2usize,
            any::<bool>(),
            prop::collection::vec(any::<prop::sample::Index>(), 0..4),
            1..64usize,
            0..32usize,
        ),
        1..24,
    )
    .prop_map(|raw| {
        let mut nodes: Vec<(usize, bool, Vec<usize>, usize, usize)> = Vec::new();
        for (worker, gpu, picks, len, scratch) in raw {
            let same: Vec<usize> = (0..nodes.len()).filter(|&u| nodes[u].0 == worker).collect();
            let mut inputs: Vec<usize> = if same.is_empty() {
                Vec::new()
            } else {
                picks.iter().map(|p| *p.get(&same)).collect()
            };
            inputs.sort_unstable();
            inputs.dedup();
            nodes.push((worker, gpu, inputs, len, scratch));
        }
        Spec { nodes }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any DAG, on two stream workers and three CPU workers, twice: every
    /// read sees exactly what its producer wrote, every node runs once a
    /// run, and no view ever finds its bytes held by another. A missing read
    /// or handoff dependency shows as a wrong value or a borrow panic.
    #[test]
    fn every_dag_reads_what_was_written(spec in arb_spec()) {
        let ran = Arc::new(AtomicUsize::new(0));
        let (dag, _) = spec.dag(&ran);
        let mut instance = dag.plan().instantiate(&[0, 0]);
        for _ in 0..2 {
            prop_assert_eq!(instance.run(&mut HostOnly::new(2), 3).result, Ok(()));
        }
        prop_assert_eq!(ran.load(Ordering::Relaxed), 2 * spec.nodes.len());
    }
}

/// Writes nothing, and fails.
struct Fails;

impl Task<Heap, ()> for Fails {
    type Rows = RowId<Heap, u64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> Self::Rows {
        h.output("never", 1)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Heap>,
        _: Lane<'_, ()>,
    ) -> Status {
        Err(Error::numerical("diverged"))
    }
}

/// A failed task fails the run with its name on the error, and a task that
/// reads its output never runs.
#[test]
fn a_failure_skips_its_readers() {
    let ran = Arc::new(AtomicUsize::new(0));
    let mut dag = Dag::new();
    let never = dag.add("fails", 0, Fails);
    dag.add(
        "reader",
        0,
        Node::<true> {
            pos: 1,
            inputs: vec![(never, 0)],
            len: 1,
            scratch: 0,
            expect: Arc::new(vec![vec![0], vec![0]]),
            ran: Arc::clone(&ran),
        },
    );
    let record = dag.plan().instantiate(&[0]).run(&mut HostOnly::new(1), 1);
    let names: Vec<_> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        ["fails"],
        "the failed task is timed, its reader skipped"
    );
    let e = record.result.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Numerical);
    assert!(
        e.context.iter().any(|c| c.starts_with("fails@cpu")),
        "{:?}",
        e.context
    );
    assert_eq!(ran.load(Ordering::Relaxed), 0);
}

/// Reaches for a row it did not declare.
struct Sneaky(RowId<Heap, u64>);

impl Task<Heap, ()> for Sneaky {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, _: &mut Layout<'_, Heap>) {}

    fn run(&self, _: &(), _: &Bufs<'_, Device>, h: &Bufs<'_, Heap>, _: Lane<'_, ()>) -> Status {
        h.get(self.0);
        Ok(())
    }
}

/// What the planner refuses: a CPU task with a device row, and a read of
/// another worker's row, at `add`; and a row a task did not declare, when
/// it asks for it.
#[test]
fn refusals() {
    let m = panic_message(|| {
        let mut dag = Dag::<Heap, ()>::new();
        dag.add("cpu", 0, Declares::<false>(vec![Decl::Out("D", 1)]));
    });
    assert!(m.contains("cpu: a CPU task holds no device row"), "{m}");

    let m = panic_message(|| {
        let mut dag = Dag::<Heap, ()>::new();
        let s = dag.add("s", 0, Declares::<true>(vec![Decl::Out("S", 1)]))[0];
        dag.add("x", 1, Declares::<true>(vec![Decl::Read(s)]));
    });
    assert!(m.contains("a row of another table"), "{m}");

    let ran = Arc::new(AtomicUsize::new(0));
    let spec = Spec {
        nodes: vec![(0, false, vec![], 4, 4)],
    };
    let (mut dag, outs) = spec.dag(&ran);
    dag.add("sneaky", 0, Sneaky(outs[0]));
    let mut instance = dag.plan().instantiate(&[0]);
    let m = panic_message(|| {
        let _ = instance.run(&mut HostOnly::new(0), 1);
    });
    assert!(m.contains("node 1: out is not in its layout"), "{m}");
}

/// Two workers' chains are two tables, each the size of its chain alone,
/// with no dependency between them.
#[test]
fn two_workers_are_independent() {
    // Worker w's chain starts at position `first`.
    let chain = |worker, first| -> Vec<(usize, bool, Vec<usize>, usize, usize)> {
        vec![
            (worker, true, vec![], 32, 8),
            (worker, false, vec![first], 16, 0),
            (worker, true, vec![first + 1], 32, 0),
        ]
    };
    let mut nodes = chain(0, 0);
    nodes.extend(chain(1, 3));
    let ran = Arc::new(AtomicUsize::new(0));
    let (dag, _) = Spec {
        nodes: nodes.clone(),
    }
    .dag(&ran);
    let planned = dag.plan();

    let alone = Spec {
        nodes: nodes[..3].to_vec(),
    }
    .dag(&ran)
    .0
    .plan();
    assert_eq!(planned.n_workers(), 2);
    for w in 0..2 {
        assert_eq!(planned.host_table(w).total(), alone.host_table(0).total());
        assert_eq!(planned.device_table(w).total(), 0);
    }
    for pos in 3..6 {
        assert!(planned.predecessors(pos).iter().all(|&u| u >= 3));
    }
    planned
        .instantiate(&[0, 0])
        .run(&mut HostOnly::new(2), 1)
        .result
        .unwrap();
    assert_eq!(ran.load(Ordering::Relaxed), 6);
}

/// A declared BLAS workspace is a worker row of `BLAS_WORKSPACE_BYTES` in
/// the worker's device table, added once however often it is declared, no
/// task, and counted by admission: a cap one byte short of the table with
/// it is refused, naming it among the rows live at the peak.
#[test]
fn a_blas_workspace_is_a_worker_row_admission_counts() {
    use Decl::*;
    let n = 24 * 24;
    let task = || Declares::<true>(vec![Out("S", n)]);
    let mut bare = Dag::<Heap, ()>::new();
    bare.add("S", 0, task());
    let bare = bare.plan();

    let mut dag = Dag::<Heap, ()>::new();
    dag.blas_workspace(0);
    dag.add("S", 0, task());
    dag.blas_workspace(0);
    dag.blas_workspace(1);
    let planned = dag.plan();
    assert_eq!(planned.len(), 1, "declaring a workspace adds no task");
    assert_eq!(planned.n_workers(), 2);
    let row = planned.blas_workspaces[0].expect("worker 0's workspace");
    let table = planned.device_table(0);
    assert_eq!(table.n_rows(), bare.device_table(0).n_rows() + 1);
    assert_eq!(table.len(row), BLAS_WORKSPACE_BYTES);
    assert_eq!(
        table.total(),
        bare.device_table(0).total() + BLAS_WORKSPACE_BYTES
    );
    assert!(planned.blas_workspaces[1].is_some());
    assert_eq!(planned.device_table(1).total(), BLAS_WORKSPACE_BYTES);
    let text = planned.to_string();
    assert_eq!(text.matches("blas workspace").count(), 2, "{text}");
    assert!(text.contains("worker  plan     blas workspace"), "{text}");

    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
    let refused = fb_core_arena::admit(table, table.total() - 1, "gpu0").unwrap_err();
    assert_eq!(refused.kind, ErrorKind::Allocation);
    assert!(
        refused
            .message
            .contains(&format!("blas workspace ({BLAS_WORKSPACE_BYTES} B)")),
        "{}",
        refused.message
    );
}
