//! The flow, device-free: v0.0.4's `tests/core.rs` cases that run a DAG.
//! Five fake steps over a small bus on the plain heap ([`Backing::Host`])
//! and no GPU: `a`; `b` and `c`, which read `a`; `d`, which reads `b` and
//! `c`; and `e`, which reads nothing. What each does comes from the bus,
//! so one graph serves every case.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fb_core_budget::{
    Backing, Budget, Cap, Caps, DeviceBuf, Kind, Layout, Lease, Ledger, LockId, ManagerId, Span,
};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};

use crate::{Bus, Dag, Executor, Held, Lane, Locks, Plan, Slot, Step, View, locked};

const MIB: u64 = 1 << 20;
const N: usize = 5;
const NAMES: [&str; N] = ["a", "b", "c", "d", "e"];
const ROOMY: u64 = 1 << 40;

/// What the fake steps are told to do.
#[derive(Clone, Default)]
struct Spec {
    /// The steps whose `run` fails.
    fails: Vec<usize>,
    /// The steps that declare the `EriManager`'s lock.
    manager: Vec<usize>,
    /// The steps that declare gpu0's lock.
    gpu0: Vec<usize>,
}

struct Out {
    device: DeviceBuf<f64>,
    /// How many times `run` ran.
    runs: usize,
    /// The first double of the step's scratch as each `run` found it.
    found: Vec<f64>,
    /// The address of the step's scratch on each `run`.
    address: Vec<usize>,
    /// The locks `run` was handed, as the wrapper took them.
    taken: Vec<LockId>,
}

struct TestBus {
    spec: Spec,
    ledger: Ledger,
    locks: Locks,
    slots: [Slot<Out>; N],
    /// The steps in the order their `run`s began.
    order: Mutex<Vec<usize>>,
    /// Steps inside `run` now, and the most there ever were; the same for
    /// the holders of the manager's lock.
    running: AtomicUsize,
    most: AtomicUsize,
    holding: AtomicUsize,
    most_holding: AtomicUsize,
}

impl Bus for TestBus {
    type Shared = TestBus;
    fn shared(&self) -> &TestBus {
        self
    }
    fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    fn locks(&self) -> &Locks {
        &self.locks
    }
}

/// Step `i`'s device scratch bytes: `d` and `a` the largest, on one path.
const SCRATCH: [u64; N] = [4 * MIB, 3 * MIB, 2 * MIB, 6 * MIB, MIB];

fn span(i: usize) -> (Layout, Span<f64>) {
    let mut l = Layout::new();
    let span = l.span::<f64>(SCRATCH[i] as usize / 8);
    (l, span)
}

fn budget_of(i: usize, bus: &TestBus, budget: &Budget<'_>) -> Result<Out> {
    budget.scratch(0, Kind::Device, span(i).0.bytes());
    if bus.spec.manager.contains(&i) {
        budget.locks([LockId::Manager(ManagerId::Eri)]);
    }
    if bus.spec.gpu0.contains(&i) {
        budget.locks([LockId::Gpu(0)]);
    }
    Ok(Out {
        device: budget.device(0, MIB as usize / 8),
        runs: 0,
        found: Vec::new(),
        address: Vec::new(),
        taken: Vec::new(),
    })
}

fn bump(now: &AtomicUsize, most: &AtomicUsize) {
    let n = now.fetch_add(1, Ordering::AcqRel) + 1;
    most.fetch_max(n, Ordering::AcqRel);
}

fn run_of(i: usize, out: &mut Out, bus: &TestBus, held: &Held<'_>, scratch: &Lease<'_>) -> Status {
    locked(&bus.order).push(i);
    bump(&bus.running, &bus.most);
    let holds = held.taken().contains(&LockId::Manager(ManagerId::Eri));
    if holds {
        bump(&bus.holding, &bus.most_holding);
    }
    out.taken = held.taken().to_vec();
    let mut mine = scratch.device(0, span(i).1);
    let bytes = mine.host().expect("a host-backed plan");
    out.address.push(bytes.as_ptr() as usize);
    out.found.push(bytes.first().copied().unwrap_or(0.0));
    // Fill the region with the step's own value, give an unordered step
    // time to stomp on it, and look again.
    let value = (i + 1) as f64;
    bytes.fill(value);
    std::thread::sleep(std::time::Duration::from_millis(2));
    let intact = bytes.iter().all(|&x| x == value);
    out.device.host().expect("host-backed").fill(value);
    out.runs += 1;
    if holds {
        bus.holding.fetch_sub(1, Ordering::AcqRel);
    }
    bus.running.fetch_sub(1, Ordering::AcqRel);
    assert!(
        intact,
        "step '{}' had its scratch written under it",
        NAMES[i]
    );
    if bus.spec.fails.contains(&i) {
        return Err(Error::numerical("told to fail"));
    }
    Ok(())
}

macro_rules! fake {
    ($name:ident, $index:expr, $reads:ty) => {
        #[derive(Clone, Copy)]
        struct $name;
        impl Step<TestBus> for $name {
            const NAME: &'static str = NAMES[$index];
            type Output = Out;
            type Reads = $reads;
            fn slot(bus: &TestBus) -> &Slot<Out> {
                &bus.slots[$index]
            }
            fn mem_budget(bus: &TestBus, budget: &Budget<'_>) -> Result<Out> {
                budget_of($index, bus, budget)
            }
            fn run(
                out: &mut Out,
                view: &View<'_, TestBus, Self>,
                held: &mut Held<'_>,
                _: &Lane,
                scratch: &Lease<'_>,
            ) -> Status {
                run_of($index, out, view.shared(), held, scratch)
            }
        }
    };
}

fake!(A, 0, ());
fake!(B, 1, (A,));
fake!(C, 2, (A,));
fake!(D, 3, (B, C));
fake!(E, 4, ());
// Two steps that read each other, under a's and b's names.
fake!(F, 0, (G,));
fake!(G, 1, (F,));

const PREDS: [&[usize]; N] = [&[], &[0], &[0], &[1, 2], &[]];

fn bus_with(spec: Spec, backing: Backing, nan_fill: bool) -> Arc<TestBus> {
    let cap = Cap {
        written: Some(ROOMY),
        used: ROOMY,
    };
    Arc::new(TestBus {
        spec,
        ledger: Ledger::new(
            Caps {
                device: vec![cap],
                host: cap,
            },
            backing,
            vec![0],
            nan_fill,
        ),
        locks: Locks::new(&[]),
        slots: std::array::from_fn(|_| Slot::new()),
        order: Mutex::new(Vec::new()),
        running: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
        holding: AtomicUsize::new(0),
        most_holding: AtomicUsize::new(0),
    })
}

fn bus(spec: Spec) -> Arc<TestBus> {
    bus_with(spec, Backing::Host, false)
}

/// The plan with the steps added in `order`.
fn plan_in(bus: Arc<TestBus>, order: [usize; N]) -> Plan<TestBus> {
    let mut dag = Dag::new();
    for i in order {
        match i {
            0 => dag.add(A),
            1 => dag.add(B),
            2 => dag.add(C),
            3 => dag.add(D),
            _ => dag.add(E),
        };
    }
    dag.build(bus).expect("the steps close")
}

fn plan(bus: Arc<TestBus>) -> Plan<TestBus> {
    plan_in(bus, [0, 1, 2, 3, 4])
}

fn runs(plan: &Plan<TestBus>) -> [usize; N] {
    [
        plan.output(A).runs,
        plan.output(B).runs,
        plan.output(C).runs,
        plan.output(D).runs,
        plan.output(E).runs,
    ]
}

#[test]
fn a_missing_declared_read_is_refused_when_the_dag_is_built() {
    // d reads b and c, b reads a; the config asks for neither a nor c.
    let mut dag = Dag::new();
    dag.add(D).add(B);
    let error = dag
        .build(bus(Spec::default()))
        .err()
        .expect("the steps do not close");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(
        error.message,
        "step 'd' reads step 'c', which the config does not ask for"
    );
    // With c asked for, b's read of a is the one refused.
    let mut dag = Dag::new();
    dag.add(D).add(C).add(B);
    let error = dag.build(bus(Spec::default())).err().expect("a is missing");
    assert_eq!(
        error.message,
        "step 'c' reads step 'a', which the config does not ask for"
    );
}

#[test]
fn a_cycle_of_reads_is_refused() {
    let mut dag = Dag::new();
    dag.add(F).add(G).add(E);
    let error = dag.build(bus(Spec::default())).err().expect("a cycle");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(error.message, "the steps' reads make a cycle through a, b");
}

#[test]
fn a_step_added_twice_is_refused() {
    let mut dag = Dag::new();
    dag.add(A).add(A);
    let error = dag.build(bus(Spec::default())).err().expect("a twice");
    assert_eq!(error.message, "step 'a' is in the plan twice");
}

#[test]
fn every_node_runs_once_in_a_topological_order() {
    // On eight workers, over a replay: each execute runs every node once,
    // each after every step it reads.
    for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]] {
        let plan = plan_in(bus(Spec::default()), order);
        let executor = Executor::new(8);
        plan.budget(&executor).unwrap();
        for replay in 1..=3 {
            locked(&plan.bus().order).clear();
            plan.execute(&executor).unwrap();
            let ran = locked(&plan.bus().order).clone();
            let mut sorted = ran.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                [0, 1, 2, 3, 4],
                "{order:?}, replay {replay}: {ran:?}"
            );
            let at = |s: usize| ran.iter().position(|&r| r == s).unwrap();
            for (s, preds) in PREDS.iter().enumerate() {
                for &p in *preds {
                    assert!(
                        at(p) < at(s),
                        "{order:?}: {} before {}: {ran:?}",
                        NAMES[s],
                        NAMES[p]
                    );
                }
            }
            assert_eq!(runs(&plan), [replay; N]);
        }
    }
}

#[test]
fn a_failed_step_fails_its_successors_and_nothing_else() {
    // b fails: d, which reads it, is skipped; a, c and e run.
    let plan = plan(bus(Spec {
        fails: vec![1],
        ..Spec::default()
    }));
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    let error = plan.execute(&executor).expect_err("b fails");
    assert_eq!(error.kind, ErrorKind::Numerical);
    assert_eq!(error.message, "told to fail");
    assert_eq!(error.context, ["step 'b'"]);
    assert_eq!(runs(&plan), [1, 1, 1, 0, 1]);
    // a fails: everything but e is skipped.
    let plan = self::plan(bus(Spec {
        fails: vec![0],
        ..Spec::default()
    }));
    plan.budget(&executor).unwrap();
    plan.execute(&executor).expect_err("a fails");
    assert_eq!(runs(&plan), [1, 0, 0, 0, 1]);
}

#[test]
fn the_first_error_is_the_first_in_topological_order() {
    // e and c fail; c comes first in the one order (a, b, c, d, e),
    // whichever finishes first, and e's failure is on its context.
    for workers in [1, 4, 8] {
        let plan = plan(bus(Spec {
            fails: vec![4, 2],
            ..Spec::default()
        }));
        let executor = Executor::new(workers);
        plan.budget(&executor).unwrap();
        let error = plan.execute(&executor).expect_err("c and e fail");
        assert_eq!(error.context[0], "step 'c'", "{workers} workers");
        assert!(
            error.context[1].starts_with("and also: told to fail"),
            "{error}"
        );
        assert!(error.context[1].contains("step 'e'"), "{error}");
        assert_eq!(runs(&plan), [1, 1, 1, 0, 1]);
    }
}

#[test]
fn unordered_steps_overlap_in_time_and_never_in_bytes() {
    // Every step checks its scratch was not written under it; b and c, and
    // e with everyone, are free to run at once on eight workers.
    let mut overlapped = 0;
    for _ in 0..20 {
        let plan = plan(bus(Spec::default()));
        let executor = Executor::new(8);
        plan.budget(&executor).unwrap();
        plan.execute(&executor)
            .expect("no step's scratch is stomped");
        overlapped += usize::from(plan.bus().most.load(Ordering::Acquire) > 1);
    }
    assert!(
        overlapped > 0,
        "no run had two steps at once: nothing was tested"
    );
}

#[test]
fn the_wrapper_takes_a_declared_lock_and_its_holders_never_overlap() {
    // b, c and e declare the manager's lock: unordered, they may be ready
    // together, and run one at a time; each is handed the guard.
    let plan = plan(bus(Spec {
        manager: vec![1, 2, 4],
        ..Spec::default()
    }));
    let executor = Executor::new(8);
    plan.budget(&executor).unwrap();
    for _ in 0..10 {
        plan.execute(&executor).unwrap();
    }
    assert_eq!(plan.bus().most_holding.load(Ordering::Acquire), 1);
    let eri = [LockId::Manager(ManagerId::Eri)];
    assert_eq!(plan.output(B).taken, eri);
    assert_eq!(plan.output(E).taken, eri);
    assert_eq!(plan.output(A).taken, []);
}

#[test]
fn a_lock_of_a_gpu_the_step_does_not_run_on_is_refused() {
    // a is a host step, and declares gpu0's lock.
    let plan = plan(bus(Spec {
        gpu0: vec![0],
        ..Spec::default()
    }));
    let error = plan
        .budget(&Executor::new(2))
        .expect_err("a runs on no gpu");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(
        error.message,
        "step 'a' declares the lock of gpu 0, which it does not run on"
    );
}

#[test]
fn the_budget_is_the_same_under_any_order_on_several_workers() {
    let reference = {
        let plan = plan(bus(Spec::default()));
        plan.budget(&Executor::new(1)).unwrap();
        (plan.bus().ledger.regions(), plan.bus().ledger.table())
    };
    for order in [[4, 3, 2, 1, 0], [2, 4, 0, 3, 1], [3, 0, 4, 1, 2]] {
        let plan = plan_in(bus(Spec::default()), order);
        plan.budget(&Executor::new(4)).unwrap();
        assert_eq!(plan.bus().ledger.regions(), reference.0, "{order:?}");
        assert_eq!(plan.bus().ledger.table(), reference.1, "{order:?}");
    }
}

#[test]
fn a_replay_allocates_nothing_and_keeps_its_addresses_and_nan_fill_shows_stale_reads() {
    let plan = plan(bus_with(Spec::default(), Backing::Host, true));
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    let allocations = plan.bus().ledger.allocations();
    for _ in 0..3 {
        plan.execute(&executor).unwrap();
    }
    assert_eq!(plan.bus().ledger.allocations(), allocations);
    let a = plan.output(A);
    assert_eq!(a.runs, 3);
    assert!(a.address.windows(2).all(|w| w[0] == w[1]));
    // NaN on entry every run, though the step wrote its value there the
    // run before.
    assert!(a.found.iter().all(|x| x.is_nan()), "{:?}", a.found);
    drop(a);
    // Without the fill, a step whose region nobody shares finds its old
    // data: the contents are undefined, not cleared.
    let plain = self::plan(bus(Spec::default()));
    plain.budget(&executor).unwrap();
    plain.execute(&executor).unwrap();
    plain.execute(&executor).unwrap();
    assert_eq!(plain.output(E).found, [0.0, 5.0]);
}

#[test]
fn a_dry_budget_counts_and_cannot_run() {
    let plan = plan(bus_with(Spec::default(), Backing::Dry, false));
    let executor = Executor::new(2);
    plan.budget(&executor).unwrap();
    assert_eq!(plan.bus().ledger.allocations(), 0);
    let error = plan.execute(&executor).expect_err("nothing behind it");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

#[test]
fn fan_out_returns_in_gpu_order_and_the_first_error() {
    let squares = crate::fan_out(vec![1, 2, 3, 4], |x| Ok(x * x)).unwrap();
    assert_eq!(squares, [1, 4, 9, 16]);
    let error = crate::fan_out(vec![1, 2, 3], |x| {
        if x >= 2 {
            Err(Error::numerical(format!("gpu {x}")))
        } else {
            Ok(x)
        }
    })
    .expect_err("two fail");
    assert_eq!(error.message, "gpu 2");
}
