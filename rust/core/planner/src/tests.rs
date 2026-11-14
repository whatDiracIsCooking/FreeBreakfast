//! The planner's device-free tests: tasks over the heap kind, run on
//! [`HostOnly`] workers (tier 3).

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;

use super::*;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, RowId};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_core_thread_pool::{HostOnly, LaneId, Placement};
use fb_test_support::{golden, panic_message};

/// One row declaration, in the order a [`Declares`] makes them.
#[derive(Clone, Copy)]
enum Decl {
    Read(RowId<Device, f64>),
    /// A loop's carried row, written by its last user in the body.
    Write(RowId<Device, f64>),
    Out(&'static str, usize),
    Scratch(&'static str, usize),
}

/// A GPU task that only declares device rows, in a given order, and
/// returns the ones it made. For planning tests; its run does nothing.
struct Declares(Vec<Decl>);

impl GpuTask<Heap, ()> for Declares {
    type Rows = Vec<RowId<Device, f64>>;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let mut made = Vec::new();
        for &decl in &self.0 {
            match decl {
                Decl::Read(r) => {
                    d.read(r);
                }
                Decl::Write(r) => {
                    d.write(r);
                }
                Decl::Out(name, len) => made.push(d.output(name, len)),
                Decl::Scratch(name, len) => made.push(d.scratch(name, len)),
            }
        }
        made
    }

    fn run(&self, _: &Self::Rows, _: &Bufs<'_, Device>, _: &CopyOnly<'_, Heap>, _: &()) -> Status {
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
    let s = dag.add_gpu("overlap", 0, Declares(vec![Out("S", n2)]))[0];
    let h = dag.add_gpu("core", 0, Declares(vec![Out("H", n2)]))[0];
    let x = dag.add_gpu(
        "orthogonalizer",
        0,
        Declares(vec![
            Read(s),
            Out("X", n2),
            Scratch("syevd work (X)", lwork),
        ]),
    )[0];
    let d = dag.add_gpu("guess", 0, Declares(vec![Read(h), Read(x), Out("D", n2)]))[0];
    let f = dag.add_gpu(
        "fock",
        0,
        Declares(vec![Read(h), Read(d), Scratch("J", n2), Out("F", n2)]),
    )[1];
    let c = dag.add_gpu(
        "diagonalize",
        0,
        Declares(vec![
            Read(f),
            Read(x),
            Out("C", n2),
            Scratch("syevd work (F)", lwork),
        ]),
    )[0];
    dag.add_gpu("density", 0, Declares(vec![Read(c), Out("D next", n2)]));
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

/// A decision that holds no row and ends the loop after its first
/// iteration. For planning tests.
struct Stops;

impl Decision<Heap> for Stops {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Heap>) {}

    fn decide(&self, _: &(), _: &Bufs<'_, Heap>, _: usize) -> Result<ControlFlow<()>> {
        Ok(ControlFlow::Break(()))
    }
}

/// `arena::tests::scf_loop_fixture` declared as tasks: the SCF iteration's
/// tasks as a loop's body, after the tasks that make S, H, X and the guess,
/// the density writing the D the Fock build reads.
fn scf_loop_dag(nbf: usize) -> Dag<'static, Heap, ()> {
    use Decl::*;
    let n2 = nbf * nbf;
    let lwork = 1 + 6 * nbf + 2 * n2;
    let mut dag = Dag::new();
    let _info = dag.device_worker_row::<i32>(0, "solver info", 1);
    let s = dag.add_gpu("overlap", 0, Declares(vec![Out("S", n2)]))[0];
    let h = dag.add_gpu("core", 0, Declares(vec![Out("H", n2)]))[0];
    let x = dag.add_gpu("orthogonalizer", 0, Declares(vec![Read(s), Out("X", n2)]))[0];
    let d = dag.add_gpu("guess", 0, Declares(vec![Read(h), Read(x), Out("D", n2)]))[0];
    let body = dag.len();
    let f = dag.add_gpu(
        "fock",
        0,
        Declares(vec![Read(h), Read(d), Scratch("J", n2), Out("F", n2)]),
    )[1];
    dag.add_gpu(
        "residual",
        0,
        Declares(vec![Read(f), Read(d), Read(s), Read(x), Out("e", n2)]),
    );
    let c = dag.add_gpu(
        "diagonalize",
        0,
        Declares(vec![
            Read(f),
            Read(x),
            Out("C", n2),
            Scratch("syevd work (F)", lwork),
        ]),
    )[0];
    dag.add_gpu("density", 0, Declares(vec![Read(c), Write(d)]));
    dag.add_loop("converged", 0, body, 50, Stops);
    dag
}

/// The loop's tables are the arena's loop golden byte for byte, its body
/// in the region; the decision runs after every body task; and the whole
/// plan, the loop on its decision's line, is a golden of its own.
#[test]
fn scf_loop_plan_golden() {
    let planned = scf_loop_dag(24).plan();
    assert_eq!(
        planned.device_table(0).to_string(),
        include_str!("../../arena/tests/golden/arena_scf_loop.txt")
    );
    assert_eq!(planned.predecessors(8), &[4, 5, 6, 7]);
    assert_eq!(planned.predecessors(7), &[3, 4, 5, 6]);
    assert_eq!(planned.placement(8), Placement::Cpu);
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "planner_scf_loop.txt",
        &planned.to_string(),
    );
}

/// The iterate `x <- x / 2 + 1` from 0, a loop's body over heap rows: the
/// step reads the carried `x` and writes `next`, a body row; the carry
/// writes `next` back into `x`; the decision compares `next` with `prev`,
/// a row it carries itself, and stops when they agree to `1e-12`. The
/// `k`-th iteration moves `2^(1-k)`, so the 41st is the first to stop.
struct Iterate;

#[derive(Clone, Copy)]
struct IterateRows {
    x: RowId<Heap, f64>,
    prev: RowId<Heap, f64>,
}

impl CpuTask<Heap> for Iterate {
    type Rows = IterateRows;

    fn layout(&self, h: &mut Layout<'_, Heap>) -> IterateRows {
        IterateRows {
            x: h.output("x", 1),
            prev: h.output("prev", 1),
        }
    }

    fn run(&self, r: &IterateRows, h: &Bufs<'_, Heap>) -> Status {
        h.take_mut(r.x)[0] = 0.0;
        h.take_mut(r.prev)[0] = 0.0;
        Ok(())
    }
}

/// `next = x / 2 + 1`, on the stream.
struct Step(RowId<Heap, f64>);

impl GpuTask<Heap, ()> for Step {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> Self::Rows {
        h.read(self.0);
        h.output("next", 1)
    }

    fn run(
        &self,
        &next: &Self::Rows,
        _: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Heap>,
        _: &(),
    ) -> Status {
        let h = h.bufs();
        h.take_mut(next)[0] = h.get(self.0)[0] / 2.0 + 1.0;
        Ok(())
    }
}

/// `x = next`: the carried row's writer.
struct Carry {
    next: RowId<Heap, f64>,
    x: RowId<Heap, f64>,
}

impl CpuTask<Heap> for Carry {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Heap>) {
        h.read(self.next);
        h.write(self.x);
    }

    fn run(&self, _: &(), h: &Bufs<'_, Heap>) -> Status {
        h.take_mut(self.x)[0] = h.get(self.next)[0];
        Ok(())
    }
}

/// Stops when `next` and the `prev` it carries agree to `1e-12`.
struct Converged {
    next: RowId<Heap, f64>,
    prev: RowId<Heap, f64>,
}

impl Decision<Heap> for Converged {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Heap>) {
        h.read(self.next);
        h.write(self.prev);
    }

    fn decide(&self, _: &(), h: &Bufs<'_, Heap>, _: usize) -> Result<ControlFlow<()>> {
        let next = h.get(self.next)[0];
        let prev = h.take_mut(self.prev);
        let moved = (next - prev[0]).abs();
        prev[0] = next;
        Ok(if moved > 1e-12 {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    }
}

/// Reads `x` after the loop.
struct Report(RowId<Heap, f64>, Arc<std::sync::Mutex<f64>>);

impl CpuTask<Heap> for Report {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Heap>) {
        h.read(self.0);
    }

    fn run(&self, _: &(), h: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap() = h.get(self.0)[0];
        Ok(())
    }
}

/// A loop through the planner and the executor: the body's rows in the
/// region, `x` and `prev` carried, one by a task and one by the decision,
/// and `x` read after the loop through the decision. The fixed point is 2,
/// reached to `2^-40` in 41 iterations; the instance runs twice over the
/// same blocks, each run from the start, with no borrow ever contended.
#[test]
fn a_loop_carries_its_rows_to_the_fixed_point() {
    let out = Arc::new(std::sync::Mutex::new(0.0));
    let mut dag = Dag::<Heap, ()>::new();
    let IterateRows { x, prev } = dag.add_cpu("init", 0, Iterate);
    let body = dag.len();
    let next = dag.add_gpu("step", 0, Step(x));
    dag.add_cpu("carry", 0, Carry { next, x });
    dag.add_loop("converged", 0, body, 100, Converged { next, prev });
    dag.add_cpu("report", 0, Report(x, Arc::clone(&out)));
    let planned = dag.plan();
    assert_eq!(
        planned.predecessors(2),
        &[0, 1],
        "the carry after the step's read of x"
    );
    assert_eq!(planned.predecessors(3), &[0, 1, 2]);
    assert_eq!(
        planned.predecessors(4),
        &[0, 3],
        "x after the loop through its decision"
    );
    let text = planned.to_string();
    assert!(text.contains("loop 1-3: region of"), "{text}");
    assert!(
        text.contains("decides loop 1-3, at most 100 iterations"),
        "{text}"
    );

    let mut instance = planned.instantiate(&[0]);
    for _ in 0..2 {
        let record = instance.run(&mut HostOnly::new(1), 2);
        record.result.unwrap();
        let decision = record
            .iterations
            .iter()
            .map(|(&node, &n)| (node.index(), n))
            .collect::<Vec<_>>();
        assert_eq!(decision, [(3, 41)]);
        let got = *out.lock().unwrap();
        assert!((got - 2.0).abs() <= 1e-12, "{got}");
    }
}

/// A loop inside a loop's body is refused when it is added.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a body holds no loop")]
fn a_body_holds_no_loop() {
    let mut dag = Dag::<Heap, ()>::new();
    let body = dag.len();
    dag.add_gpu("inner", 0, Declares(vec![]));
    dag.add_loop("inner loop", 0, body, 2, Stops);
    dag.add_loop("outer loop", 0, body, 2, Stops);
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

impl<const GPU: bool> Node<GPU> {
    /// Either sort's layout.
    fn declare(&self, h: &mut Layout<'_, Heap>) -> NodeRows {
        for &(row, _) in &self.inputs {
            h.read(row);
        }
        NodeRows {
            scratch: h.scratch("scratch", self.scratch),
            out: h.output("out", self.len),
        }
    }

    /// Either sort's run.
    fn compute(&self, rows: &NodeRows, h: &Bufs<'_, Heap>) -> Status {
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

impl GpuTask<Heap, ()> for Node<true> {
    type Rows = NodeRows;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> NodeRows {
        self.declare(h)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len), ("scratch", self.scratch)]
    }

    fn run(&self, rows: &NodeRows, _: &Bufs<'_, Device>, h: &CopyOnly<'_, Heap>, _: &()) -> Status {
        self.compute(rows, h.bufs())
    }
}

impl CpuTask<Heap> for Node<false> {
    type Rows = NodeRows;

    fn layout(&self, h: &mut Layout<'_, Heap>) -> NodeRows {
        self.declare(h)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len), ("scratch", self.scratch)]
    }

    fn run(&self, rows: &NodeRows, h: &Bufs<'_, Heap>) -> Status {
        self.compute(rows, h)
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

/// A DAG of [`Node`]s, each `(at, gpu, inputs, len, scratch)`, its
/// reference computed serially: `at` is the node's worker and, for a GPU
/// node, its lane.
#[derive(Debug)]
struct Spec {
    nodes: Vec<(LaneId, bool, Vec<usize>, usize, usize)>,
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
        for (pos, (at, gpu, inputs, len, scratch)) in self.nodes.iter().enumerate() {
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
                dag.add_gpu(name, *at, node!(true))
            } else {
                dag.add_cpu(name, at.worker, node!(false))
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
            (0.into(), false, vec![], N, 0),
            (0.into(), true, vec![0], N, 0),
            (0.into(), false, vec![0, 1], N, 0),
            (0.into(), true, vec![2], N, 0),
            (0.into(), false, vec![3], N, 0),
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
    assert_eq!(planned.placement(1), Placement::Stream(0.into()));
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
            0..3usize,
            any::<bool>(),
            prop::collection::vec(any::<prop::sample::Index>(), 0..4),
            1..64usize,
            0..32usize,
        ),
        1..24,
    )
    .prop_map(|raw| {
        let mut nodes: Vec<(LaneId, bool, Vec<usize>, usize, usize)> = Vec::new();
        for (worker, lane, gpu, picks, len, scratch) in raw {
            let lane = if gpu { lane } else { 0 };
            let same: Vec<usize> = (0..nodes.len())
                .filter(|&u| nodes[u].0.worker == worker)
                .collect();
            let mut inputs: Vec<usize> = if same.is_empty() {
                Vec::new()
            } else {
                picks.iter().map(|p| *p.get(&same)).collect()
            };
            inputs.sort_unstable();
            inputs.dedup();
            nodes.push((LaneId { worker, lane }, gpu, inputs, len, scratch));
        }
        Spec { nodes }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any DAG, on two GPU workers of up to three lanes each and three CPU
    /// workers, twice: every read sees exactly what its producer wrote,
    /// across lanes as within one, every node runs once a run, and no view
    /// ever finds its bytes held by another. A missing read or handoff
    /// dependency shows as a wrong value or a borrow panic.
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

/// Row `k` of a random loop's parent, as its producer writes it.
fn parent_value(k: usize, len: usize) -> Vec<u64> {
    (0..len).map(|i| mix(0, i, &[&[k as u64 + 1]])).collect()
}

/// A carried row's next value: its writer at `pos` mixes the value the
/// iteration before left with what it reads.
fn carried_value(pos: usize, old: &[u64], inputs: &[&[u64]]) -> Vec<u64> {
    let mut all = vec![old];
    all.extend_from_slice(inputs);
    (0..old.len()).map(|i| mix(pos + 1000, i, &all)).collect()
}

/// The parent's rows of a random loop, made before it.
struct Parent(Vec<usize>);

impl CpuTask<Heap> for Parent {
    type Rows = Vec<RowId<Heap, u64>>;

    fn layout(&self, h: &mut Layout<'_, Heap>) -> Self::Rows {
        self.0.iter().map(|&len| h.output("parent", len)).collect()
    }

    fn run(&self, rows: &Self::Rows, h: &Bufs<'_, Heap>) -> Status {
        for (k, &row) in rows.iter().enumerate() {
            h.take_mut(row).copy_from_slice(&parent_value(k, self.0[k]));
        }
        Ok(())
    }
}

/// What a random loop's body node adds to its position per iteration, in
/// what its output mixes in.
const ITERATION: usize = 100;

/// A node of a random loop's body: reads its rows, writes the row it
/// carries from them and the value the iteration before left, writes its
/// scratch with junk and its output from what it read.
struct BodyNode<const GPU: bool> {
    pos: usize,
    reads: Vec<RowId<Heap, u64>>,
    carries: Option<RowId<Heap, u64>>,
    len: usize,
    scratch: usize,
}

impl<const GPU: bool> BodyNode<GPU> {
    /// Either sort's layout.
    fn declare(&self, h: &mut Layout<'_, Heap>) -> NodeRows {
        for &row in &self.reads {
            h.read(row);
        }
        if let Some(row) = self.carries {
            h.write(row);
        }
        NodeRows {
            scratch: h.scratch("scratch", self.scratch),
            out: h.output("out", self.len),
        }
    }

    /// Either sort's run: in iteration `k`, its output mixes `k` in.
    fn compute(&self, rows: &NodeRows, h: &Bufs<'_, Heap>) -> Status {
        let ins: Vec<&[u64]> = self.reads.iter().map(|&r| h.get(r)).collect();
        if let Some(row) = self.carries {
            let carried = h.take_mut(row);
            let next = carried_value(self.pos, carried, &ins);
            carried.copy_from_slice(&next);
        }
        h.take_mut(rows.scratch).fill(u64::MAX - self.pos as u64);
        let out = h.take_mut(rows.out);
        let at = self.pos + ITERATION * h.iteration();
        for (i, x) in out.iter_mut().enumerate() {
            *x = mix(at, i, &ins);
        }
        Ok(())
    }
}

impl GpuTask<Heap, ()> for BodyNode<true> {
    type Rows = NodeRows;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> NodeRows {
        self.declare(h)
    }

    fn run(&self, rows: &NodeRows, _: &Bufs<'_, Device>, h: &CopyOnly<'_, Heap>, _: &()) -> Status {
        self.compute(rows, h.bufs())
    }
}

impl CpuTask<Heap> for BodyNode<false> {
    type Rows = NodeRows;

    fn layout(&self, h: &mut Layout<'_, Heap>) -> NodeRows {
        self.declare(h)
    }

    fn run(&self, rows: &NodeRows, h: &Bufs<'_, Heap>) -> Status {
        self.compute(rows, h)
    }
}

/// A random loop's decision: writes the row it carries, if any, and stops
/// after `iterations`.
struct Count {
    pos: usize,
    carries: Option<RowId<Heap, u64>>,
    iterations: usize,
}

impl Decision<Heap> for Count {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Heap>) {
        if let Some(row) = self.carries {
            h.write(row);
        }
    }

    fn decide(&self, _: &(), h: &Bufs<'_, Heap>, iteration: usize) -> Result<ControlFlow<()>> {
        if let Some(row) = self.carries {
            let carried = h.take_mut(row);
            let next = carried_value(self.pos, carried, &[]);
            carried.copy_from_slice(&next);
        }
        Ok(if iteration + 1 < self.iterations {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    }
}

/// Rows' values by the model's index, as the node after a random loop
/// leaves them.
type Collected = Arc<std::sync::Mutex<Vec<(usize, Vec<u64>)>>>;

/// After a random loop: copies out the rows it reads, by the model's
/// index.
struct Collect(Vec<(usize, RowId<Heap, u64>)>, Collected);

impl CpuTask<Heap> for Collect {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Heap>) {
        for &(_, row) in &self.0 {
            h.read(row);
        }
    }

    fn run(&self, _: &(), h: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap() = self
            .0
            .iter()
            .map(|&(k, r)| (k, h.get(r).to_vec()))
            .collect();
        Ok(())
    }
}

/// One body node of a [`LoopSpec`]: where it runs, the rows it reads by
/// the model's index (the parent's first, then the body's outputs), the
/// parent row it carries, its output's and scratch's lengths, and whether
/// its output is read after the loop.
#[derive(Clone, Debug)]
struct BodySpec {
    gpu: bool,
    reads: Vec<usize>,
    carries: Option<usize>,
    len: usize,
    scratch: usize,
    after: bool,
}

/// A loop between a node that makes the parent's rows and one that reads
/// some of them, and some of the body's outputs, after it: the body's
/// shape, the rows it and its decision carry, and its iteration count.
/// A counted loop's (`depth` above 0) carried rows and outputs read after
/// it have one user in the body each, and it has no decision.
#[derive(Debug)]
struct LoopSpec {
    parent: Vec<usize>,
    body: Vec<BodySpec>,
    decision_carries: Option<usize>,
    /// Which parent rows the node after the loop reads.
    after: Vec<bool>,
    iterations: usize,
    max_iterations: usize,
    /// A counted loop's depth, or 0 for a decision loop.
    depth: usize,
}

impl LoopSpec {
    /// The rows the node after the loop reads, by the model's index.
    fn collected(&self) -> Vec<usize> {
        let p = self.parent.len();
        (0..p)
            .filter(|&k| self.after[k])
            .chain(
                (0..self.body.len())
                    .filter(|&j| self.body[j].after)
                    .map(|j| p + j),
            )
            .collect()
    }

    /// The model: the loop run serially, its rows' values after it.
    fn expect(&self) -> Vec<(usize, Vec<u64>)> {
        let p = self.parent.len();
        let mut rows: Vec<Vec<u64>> = (0..p).map(|k| parent_value(k, self.parent[k])).collect();
        rows.extend(self.body.iter().map(|b| vec![0; b.len]));
        let decision = self.body.len() + 1;
        for it in 0..self.iterations {
            for (j, b) in self.body.iter().enumerate() {
                let ins: Vec<Vec<u64>> = b.reads.iter().map(|&k| rows[k].clone()).collect();
                let ins: Vec<&[u64]> = ins.iter().map(Vec::as_slice).collect();
                if let Some(c) = b.carries {
                    rows[c] = carried_value(j + 1, &rows[c], &ins);
                }
                let at = j + 1 + ITERATION * it;
                rows[p + j] = (0..b.len).map(|i| mix(at, i, &ins)).collect();
            }
            if let Some(c) = self.decision_carries {
                rows[c] = carried_value(decision, &rows[c], &[]);
            }
        }
        self.collected()
            .into_iter()
            .map(|k| (k, rows[k].clone()))
            .collect()
    }

    /// The DAG, and where the node after the loop leaves what it read.
    fn dag(&self) -> (Dag<'static, Heap, ()>, Collected) {
        let mut dag = Dag::new();
        let mut rows = dag.add_cpu("parent", 0, Parent(self.parent.clone()));
        let body = dag.len();
        for (j, b) in self.body.iter().enumerate() {
            let pos = dag.len();
            let reads = b.reads.iter().map(|&k| rows[k]).collect();
            let carries = b.carries.map(|c| rows[c]);
            macro_rules! node {
                ($gpu:literal) => {
                    BodyNode::<$gpu> {
                        pos,
                        reads,
                        carries,
                        len: b.len,
                        scratch: b.scratch,
                    }
                };
            }
            let name = format!("b{j}");
            let made = if b.gpu {
                dag.add_gpu(name, 0, node!(true))
            } else {
                dag.add_cpu(name, 0, node!(false))
            };
            rows.push(made.out);
        }
        if self.depth > 0 {
            dag.add_counted(body, self.iterations, self.depth);
        } else {
            let decision = Count {
                pos: dag.len(),
                carries: self.decision_carries.map(|c| rows[c]),
                iterations: self.iterations,
            };
            dag.add_loop("count", 0, body, self.max_iterations, decision);
        }
        let out = Arc::new(std::sync::Mutex::new(Vec::new()));
        let collected = self.collected().into_iter().map(|k| (k, rows[k])).collect();
        dag.add_cpu("collect", 0, Collect(collected, Arc::clone(&out)));
        (dag, out)
    }
}

/// A random loop: up to four parent rows, up to five body nodes each
/// reading up to three earlier rows and carrying at most one parent row,
/// the decision perhaps carrying one more, and one to five iterations;
/// or, a third of the time, a counted loop of depth one to three (at most
/// its trip count) with no decision, each row it carries or leaves read by
/// no other body node. Lengths are small, so most rows pad to one
/// alignment and any dead one's bytes fit the next.
fn arb_loop() -> impl Strategy<Value = LoopSpec> {
    let node = (
        any::<bool>(),
        prop::collection::vec(any::<prop::sample::Index>(), 0..4),
        prop::option::of(any::<prop::sample::Index>()),
        1..9usize,
        0..9usize,
        any::<bool>(),
    );
    (
        prop::collection::vec((1..9usize, any::<bool>()), 1..5),
        prop::collection::vec(node, 1..6),
        prop::option::of(any::<prop::sample::Index>()),
        1..6usize,
        0..3usize,
        0..9usize,
    )
        .prop_map(|(parent, raw, decision, iterations, slack, depth)| {
            // Depth 0 to 5 a decision loop, 6 to 8 a counted one.
            let depth = depth.saturating_sub(5).min(iterations);
            let p = parent.len();
            // Each parent row's writer, by body index; the decision's is
            // the body's length.
            let mut writer: Vec<Option<usize>> = vec![None; p];
            let mut body = Vec::new();
            for (j, (gpu, picks, carry, len, scratch, after)) in raw.into_iter().enumerate() {
                let carries = carry.map(|c| c.index(p)).filter(|&c| writer[c].is_none());
                if let Some(c) = carries {
                    writer[c] = Some(j);
                }
                // A carried row is read only before its writer, which is
                // its last user in the loop.
                let readable: Vec<usize> = (0..p)
                    .filter(|&k| writer[k].is_none())
                    .chain(p..p + j)
                    .collect();
                let mut reads: Vec<usize> = if readable.is_empty() {
                    Vec::new()
                } else {
                    picks.iter().map(|i| *i.get(&readable)).collect()
                };
                reads.sort_unstable();
                reads.dedup();
                body.push(BodySpec {
                    gpu,
                    reads,
                    carries,
                    len,
                    scratch,
                    after,
                });
            }
            let decision_carries = decision
                .map(|c| c.index(p))
                .filter(|&c| writer[c].is_none() && depth == 0);
            if depth > 0 {
                // A counted loop's row it writes and does not own has one
                // user in the body: no other node reads a carried row, or
                // an output read after the loop.
                let lone: Vec<usize> = (0..p)
                    .filter(|&k| writer[k].is_some())
                    .chain((0..body.len()).filter(|&j| body[j].after).map(|j| p + j))
                    .collect();
                for b in &mut body {
                    b.reads.retain(|k| !lone.contains(k));
                }
            }
            LoopSpec {
                after: parent.iter().map(|&(_, a)| a).collect(),
                parent: parent.into_iter().map(|(len, _)| len).collect(),
                body,
                decision_carries,
                iterations,
                max_iterations: iterations + slack,
                depth,
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Any loop, decision or counted, through the planner and the
    /// executor, twice over the same blocks: the rows it carries and its
    /// outputs, read after it, are the serial model's after its iteration
    /// count, and a decision ran that many times. A row the layout lets
    /// another take the bytes of while the loop still needs it, a carried
    /// row or one the body reads every iteration, shows as a wrong value;
    /// a missing dependency, as a wrong value or a borrow panic. A counted
    /// loop runs its iterations overlapped, up to its depth, each in its
    /// own copy of the body's rows: a copy two iterations share, or an
    /// iteration started before the one `depth` back has finished, shows
    /// the same way.
    #[test]
    fn every_loop_carries_what_the_model_does(spec in arb_loop()) {
        let (dag, out) = spec.dag();
        let decision = spec.body.len() + 1;
        let mut instance = dag.plan().instantiate(&[0]);
        let expect = spec.expect();
        for _ in 0..2 {
            let record = instance.run(&mut HostOnly::new(1), 3);
            prop_assert_eq!(record.result, Ok(()));
            let iterations: Vec<_> = record
                .iterations
                .iter()
                .map(|(&node, &n)| (node.index(), n))
                .collect();
            let decided = if spec.depth > 0 { vec![] } else { vec![(decision, spec.iterations)] };
            prop_assert_eq!(iterations, decided);
            prop_assert_eq!(&*out.lock().unwrap(), &expect);
        }
    }
}

/// Writes nothing, and fails.
struct Fails;

impl CpuTask<Heap> for Fails {
    type Rows = RowId<Heap, u64>;

    fn layout(&self, h: &mut Layout<'_, Heap>) -> Self::Rows {
        h.output("never", 1)
    }

    fn run(&self, _: &Self::Rows, _: &Bufs<'_, Heap>) -> Status {
        Err(Error::numerical("diverged"))
    }
}

/// A failed task fails the run with its name on the error, and a task that
/// reads its output never runs.
#[test]
fn a_failure_skips_its_readers() {
    let ran = Arc::new(AtomicUsize::new(0));
    let mut dag = Dag::new();
    let never = dag.add_cpu("fails", 0, Fails);
    dag.add_gpu(
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

impl CpuTask<Heap> for Sneaky {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Heap>) {}

    fn run(&self, _: &(), h: &Bufs<'_, Heap>) -> Status {
        h.get(self.0);
        Ok(())
    }
}

/// A read of another worker's row is refused at `add`, by the table's
/// contract.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
fn a_read_of_another_workers_row_is_refused() {
    let m = panic_message(|| {
        let mut dag = Dag::<Heap, ()>::new();
        let s = dag.add_gpu("s", 0, Declares(vec![Decl::Out("S", 1)]))[0];
        dag.add_gpu("x", 1, Declares(vec![Decl::Read(s)]));
    });
    assert!(m.contains("a row of this table"), "{m}");
}

/// What the planner refuses in every build: a row a task did not declare,
/// when it asks for it. (A CPU task with a device row does not build:
/// [`CpuTask`]'s `compile_fail` example.)
#[test]
fn refusals() {
    let ran = Arc::new(AtomicUsize::new(0));
    let spec = Spec {
        nodes: vec![(0.into(), false, vec![], 4, 4)],
    };
    let (mut dag, outs) = spec.dag(&ran);
    dag.add_cpu("sneaky", 0, Sneaky(outs[0]));
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
    let chain = |worker: usize, first| -> Vec<(LaneId, bool, Vec<usize>, usize, usize)> {
        vec![
            (worker.into(), true, vec![], 32, 8),
            (worker.into(), false, vec![first], 16, 0),
            (worker.into(), true, vec![first + 1], 32, 0),
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
    let task = || Declares(vec![Out("S", n)]);
    let mut bare = Dag::<Heap, ()>::new();
    bare.add_gpu("S", 0, task());
    let bare = bare.plan();

    let mut dag = Dag::<Heap, ()>::new();
    dag.blas_workspace(0);
    dag.add_gpu("S", 0, task());
    dag.blas_workspace(0);
    dag.blas_workspace(1);
    let planned = dag.plan();
    assert_eq!(planned.len(), 1, "declaring a workspace adds no task");
    assert_eq!(planned.n_workers(), 2);
    let (lane, row) = planned.blas_workspaces[0][0];
    assert_eq!(lane, 0);
    let table = planned.device_table(0);
    assert_eq!(table.n_rows(), bare.device_table(0).n_rows() + 1);
    assert_eq!(table.len(row), BLAS_WORKSPACE_BYTES);
    assert_eq!(
        table.total(),
        bare.device_table(0).total() + BLAS_WORKSPACE_BYTES
    );
    assert_eq!(planned.blas_workspaces[1].len(), 1);
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

/// Lanes of one worker share its tables: a task on lane 1 reads a row a
/// task on lane 0 wrote, through the same table, ordered by the read; a
/// later row on lane 0 takes the bytes of one lane 1 last used, ordered by
/// the handoff from lane 1. Each lane gets a BLAS workspace row of its own,
/// lane 0's under its one-lane name, and the plan's text names each lane.
#[test]
fn lanes_share_their_workers_tables() {
    use Decl::*;
    let n = 24 * 24;
    let lane = |lane| LaneId { worker: 0, lane };
    let mut dag = Dag::<Heap, ()>::new();
    dag.blas_workspace(0);
    let a = dag.add_gpu("A", lane(0), Declares(vec![Out("a", n)]))[0];
    let b = dag.add_gpu("B", lane(1), Declares(vec![Read(a), Out("b", n)]))[0];
    let c = dag.add_gpu("C", lane(0), Declares(vec![Read(b), Out("c", n)]))[0];
    let d = dag.add_gpu("D", lane(2), Declares(vec![Read(c), Out("d", n)]))[0];
    let planned = dag.plan();
    assert_eq!((planned.n_workers(), planned.n_lanes(0)), (1, 3));
    assert_eq!(planned.placement(1), Placement::Stream(lane(1)));
    let table = planned.device_table(0);
    assert_eq!(table.offset(c), table.offset(a), "C takes A's bytes");
    assert_eq!(table.offset(d), table.offset(b), "D takes B's bytes");
    assert_eq!(planned.predecessors(1), &[0]);
    assert_eq!(planned.predecessors(2), &[0, 1], "the handoff from lane 1");
    assert_eq!(planned.predecessors(3), &[1, 2]);

    let rows = &planned.blas_workspaces[0];
    assert_eq!(rows.len(), 3, "a workspace per lane");
    let text = planned.to_string();
    for name in [
        "blas workspace\n",
        "blas workspace lane 1\n",
        "blas workspace lane 2\n",
    ] {
        assert!(text.contains(name), "{name:?} in {text}");
    }
    for line in [
        "0       0  stream0  A",
        "1       0  stream1  B: after 0",
        "3       0  stream2  D",
    ] {
        assert!(text.contains(line), "{line:?} in {text}");
    }
    let total = planned.device_table(0).total();
    assert!(total >= 3 * BLAS_WORKSPACE_BYTES, "{total}");
}

/// A task that only launches kernels or copies, as `Declares` would.
struct NoLibraries(Declares);

impl GpuTask<Heap, ()> for NoLibraries {
    type Rows = Vec<RowId<Device, f64>>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Heap>) -> Self::Rows {
        self.0.layout(d, h)
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(&self, r: &Self::Rows, d: &Bufs<'_, Device>, h: &CopyOnly<'_, Heap>, s: &()) -> Status {
        self.0.run(r, d, h, s)
    }
}

/// A lane with no library user has no BLAS workspace row and no library
/// handles: of four lanes, lane 1's tasks all say they call no library and
/// lane 3's only task does, so the plan has workspace rows for lanes 0 and
/// 2 alone, and marks only those two for handles. Lane 0 keeps both even
/// when its one task says no; a lane with one library user among others
/// that are not keeps them.
#[test]
fn a_lane_with_no_library_user_has_no_workspace_row_or_handles() {
    use Decl::*;
    let n = 24 * 24;
    let lane = |lane| LaneId { worker: 0, lane };
    let none = |name| NoLibraries(Declares(vec![Out(name, n)]));
    let mut dag = Dag::<Heap, ()>::new();
    dag.blas_workspace(0);
    dag.add_gpu("A", lane(0), none("a"));
    dag.add_gpu("B", lane(1), none("b"));
    dag.add_gpu("B'", lane(1), none("b'"));
    dag.add_gpu("C", lane(2), none("c"));
    dag.add_gpu("C'", lane(2), Declares(vec![Out("c'", n)]));
    dag.add_gpu("D", lane(3), none("d"));
    let planned = dag.plan();
    assert_eq!(planned.n_lanes(0), 4);
    assert_eq!(planned.libraries, [vec![true, false, true, false]]);
    let lanes: Vec<usize> = planned.blas_workspaces[0].iter().map(|&(l, _)| l).collect();
    assert_eq!(lanes, [0, 2]);
    let text = planned.to_string();
    assert_eq!(text.matches("blas workspace").count(), 2, "{text}");
    assert!(text.contains("blas workspace lane 2\n"), "{text}");
    let total = planned.device_table(0).total();
    assert!(total < 3 * BLAS_WORKSPACE_BYTES, "{total}");
}
