//! The core with no device: a small bus of its own and fake steps that
//! budget heap alone. Accept and refuse are deterministic, the table is
//! complete on a refusal, a config that does not close is refused, a failed
//! step fails its readers and nothing else, a looping step is a loop of the
//! graph, and execute allocates nothing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_v02::budget::{Backing, Budget, Caps, HeapSlab, Kind, Ledger};
use fb_v02::flow::{Bus, Dag, Next, Placement, Plan, Slot, Step, View};
use rustflow::Executor;

/// What the fake steps share: how many trips the loop makes, whether the
/// failing step fails, and the order the bodies ran in.
struct Shared {
    trips: usize,
    fail: bool,
    ran: Mutex<Vec<&'static str>>,
    entered: AtomicUsize,
    left: AtomicUsize,
}

struct TestBus {
    shared: Shared,
    ledger: Ledger,
    a: Slot<Vec1, ()>,
    b: Slot<Vec1, ()>,
    c: Slot<Vec1, ()>,
    looping: Slot<Vec1, ()>,
    failing: Slot<Vec1, ()>,
    after: Slot<Vec1, ()>,
}

/// A fake step's `Output`: one budgeted slab.
struct Vec1(HeapSlab<f64>);

impl Bus for TestBus {
    type Shared = Shared;
    type Lane = ();
    type Guard = ();

    fn shared(&self) -> &Shared {
        &self.shared
    }
    fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    fn enter(&self, _: Placement) {
        self.shared.entered.fetch_add(1, Ordering::Relaxed);
    }
    fn make_lane(&self, _: Placement) -> Result<()> {
        Ok(())
    }
    fn leave(&self, _: &()) -> Status {
        self.shared.left.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn bind_workspace(&self, _: usize, _: u64) -> Status {
        Ok(())
    }
}

fn bus(heap_cap: u64, backing: Backing, trips: usize, fail: bool) -> Arc<TestBus> {
    let caps = Caps {
        device: vec![1 << 20],
        pinned: vec![1 << 20],
        heap: heap_cap,
        host_available: 1 << 30,
    };
    Arc::new(TestBus {
        shared: Shared {
            trips,
            fail,
            ran: Mutex::new(Vec::new()),
            entered: AtomicUsize::new(0),
            left: AtomicUsize::new(0),
        },
        ledger: Ledger::new(caps, backing),
        a: Slot::new(),
        b: Slot::new(),
        c: Slot::new(),
        looping: Slot::new(),
        failing: Slot::new(),
        after: Slot::new(),
    })
}

fn note(view_shared: &Shared, name: &'static str) {
    view_shared.ran.lock().unwrap().push(name);
}

/// 1000 doubles, 8000 bytes; reads nothing.
#[derive(Clone, Copy)]
struct A;
impl Step<TestBus> for A {
    const NAME: &'static str = "a";
    type Output = Vec1;
    type Reads = ();
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.a
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        Ok(Vec1(budget.heap(1000)))
    }
    fn run(out: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        out.0[0] = 1.0;
        note(view.shared(), Self::NAME);
        Ok(Next::Done)
    }
}

/// 2000 doubles; reads A, and declares a workspace.
#[derive(Clone, Copy)]
struct B;
impl Step<TestBus> for B {
    const NAME: &'static str = "b";
    type Output = Vec1;
    type Reads = (A,);
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.b
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        budget.workspace(0, 4096);
        Ok(Vec1(budget.heap(2000)))
    }
    fn run(out: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        out.0[0] = view.read(A).0[0] + 1.0;
        note(view.shared(), Self::NAME);
        Ok(Next::Done)
    }
}

/// 500 doubles; reads A and B, and declares a smaller workspace.
#[derive(Clone, Copy)]
struct C;
impl Step<TestBus> for C {
    const NAME: &'static str = "c";
    type Output = Vec1;
    type Reads = (A, B);
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.c
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        budget.workspace(0, 1024);
        Ok(Vec1(budget.heap(500)))
    }
    fn run(out: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        out.0[0] = view.read(A).0[0] + view.read(B).0[0];
        note(view.shared(), Self::NAME);
        Ok(Next::Done)
    }
}

/// Reads C; runs `Shared::trips` times.
#[derive(Clone, Copy)]
struct Looping;
impl Step<TestBus> for Looping {
    const NAME: &'static str = "looping";
    const LOOPS: bool = true;
    type Output = Vec1;
    type Reads = (C,);
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.looping
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        Ok(Vec1(budget.heap(1)))
    }
    fn run(out: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        if out.0[0] as usize >= view.shared().trips {
            // A replay: start again.
            out.0[0] = 0.0;
        }
        out.0[0] += 1.0;
        note(view.shared(), Self::NAME);
        Ok(if (out.0[0] as usize) < view.shared().trips {
            Next::Again
        } else {
            Next::Done
        })
    }
}

/// Reads A; fails when `Shared::fail`.
#[derive(Clone, Copy)]
struct Failing;
impl Step<TestBus> for Failing {
    const NAME: &'static str = "failing";
    type Output = Vec1;
    type Reads = (A,);
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.failing
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        Ok(Vec1(budget.heap(1)))
    }
    fn run(_: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        note(view.shared(), Self::NAME);
        if view.shared().fail {
            return Err(Error::numerical("the fake failure"));
        }
        Ok(Next::Done)
    }
}

/// Reads the failing step.
#[derive(Clone, Copy)]
struct After;
impl Step<TestBus> for After {
    const NAME: &'static str = "after";
    type Output = Vec1;
    type Reads = (Failing,);
    fn slot(bus: &TestBus) -> &Slot<Vec1, ()> {
        &bus.after
    }
    fn mem_budget(_: &Shared, _: &(), budget: &Budget<'_>) -> Result<Vec1> {
        Ok(Vec1(budget.heap(1)))
    }
    fn run(_: &mut Vec1, view: &View<'_, TestBus, Self>) -> Result<Next> {
        note(view.shared(), Self::NAME);
        Ok(Next::Done)
    }
}

fn all(bus: Arc<TestBus>) -> Plan<TestBus> {
    let mut dag = Dag::new();
    // Added out of order on purpose: the edges are the declarations'.
    dag.add(After)
        .add(Looping)
        .add(C)
        .add(Failing)
        .add(B)
        .add(A);
    dag.build(bus).expect("the six steps close")
}

fn position(ran: &[&str], name: &str) -> usize {
    ran.iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("{name} did not run"))
}

#[test]
fn the_edges_are_the_declared_reads() {
    let plan = all(bus(1 << 20, Backing::Real, 3, false));
    let edges = plan.edges();
    let reads = |name: &str| {
        let mut r = edges.iter().find(|(n, _)| *n == name).unwrap().1.clone();
        r.sort_unstable();
        r
    };
    assert_eq!(reads("a"), Vec::<&str>::new());
    assert_eq!(reads("b"), ["a"]);
    assert_eq!(reads("c"), ["a", "b"]);
    assert_eq!(reads("looping"), ["c"]);
    assert_eq!(reads("after"), ["failing"]);
}

#[test]
fn a_plan_within_its_caps_is_accepted_and_runs_in_edge_order() {
    let executor = Executor::new(4);
    let plan = all(bus(1 << 20, Backing::Real, 3, false));
    plan.budget(&executor).expect("accepted");
    let ledger = plan.bus().ledger();
    // 1000 + 2000 + 500 + 1 + 1 + 1 doubles.
    assert_eq!(ledger.total(Kind::Heap), 8 * 3503);
    // The workspace is a max, and the final node added it to the device.
    assert_eq!(ledger.total(Kind::Workspace(0)), 4096);
    assert_eq!(ledger.total(Kind::Device(0)), 4096);
    let allocated = ledger.allocations();

    let times = plan.execute(&executor).expect("runs");
    let ran = plan.bus().shared().ran.lock().unwrap().clone();
    assert!(position(&ran, "a") < position(&ran, "b"));
    assert!(position(&ran, "b") < position(&ran, "c"));
    assert!(position(&ran, "c") < position(&ran, "looping"));
    assert!(position(&ran, "failing") < position(&ran, "after"));
    assert_eq!(ran.iter().filter(|n| **n == "looping").count(), 3);
    assert_eq!(times.iter().find(|t| t.name == "looping").unwrap().trips, 3);
    assert_eq!(times.iter().find(|t| t.name == "a").unwrap().trips, 1);
    // The values went through the views: c = a + b = 1 + 2.
    assert_eq!(plan.output(C).0[0], 3.0);
    // Execute allocated nothing, and neither does a replay.
    assert_eq!(ledger.allocations(), allocated);
    plan.execute(&executor).expect("replays");
    assert_eq!(ledger.allocations(), allocated);
    let ran = plan.bus().shared().ran.lock().unwrap().clone();
    assert_eq!(ran.iter().filter(|n| **n == "looping").count(), 6);
    // The wrapper entered and left every body: six budgets, and per execute
    // five single runs and three trips.
    let shared = plan.bus().shared();
    assert_eq!(shared.entered.load(Ordering::Relaxed), 6 + 2 * 8);
    assert_eq!(shared.left.load(Ordering::Relaxed), 2 * 8);
}

#[test]
fn a_plan_over_its_cap_is_refused_with_the_whole_table_every_time() {
    let executor = Executor::new(4);
    let mut tables = Vec::new();
    for _ in 0..20 {
        // 28,024 bytes asked of 20,000.
        let plan = all(bus(20_000, Backing::Real, 3, false));
        let error = plan.budget(&executor).expect_err("refused");
        assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
        let ledger = plan.bus().ledger();
        assert!(ledger.over());
        // Counting went on past the cap: every step is in the table.
        assert_eq!(ledger.total(Kind::Heap), 8 * 3503);
        let table = ledger.table();
        for step in ["a", "b", "c", "looping", "failing", "after"] {
            assert!(
                table.lines().any(|l| l.starts_with(step)),
                "no row for {step}:\n{table}"
            );
        }
        assert!(
            error.message.contains(&table),
            "the refusal carries the table"
        );
        // A refused plan does not run.
        let refused = plan.execute(&executor).expect_err("not budgeted");
        assert_eq!(refused.kind, ErrorKind::InvalidConfig);
        assert!(plan.bus().shared().ran.lock().unwrap().is_empty());
        tables.push(table);
    }
    assert!(
        tables.windows(2).all(|w| w[0] == w[1]),
        "the table is the same every run"
    );
    println!("{}", tables[0]);
}

#[test]
fn a_workspace_can_refuse_a_plan_every_step_passed() {
    let executor = Executor::new(2);
    let caps = Caps {
        // Under the 4096 bytes of workspace step b declares.
        device: vec![4000],
        pinned: vec![0],
        heap: 1 << 20,
        host_available: 1 << 30,
    };
    let bus = Arc::new(TestBus {
        ledger: Ledger::new(caps, Backing::Real),
        ..Arc::try_unwrap(bus(0, Backing::Real, 1, false))
            .ok()
            .unwrap()
    });
    let plan = all(bus);
    let error = plan
        .budget(&executor)
        .expect_err("refused at the final node");
    assert_eq!(error.kind, ErrorKind::Allocation);
    assert!(error.message.contains("workspace (max)"), "{error}");
    assert!(
        error
            .message
            .contains("asks 4.00 KiB of device 0's workspace"),
        "{error}"
    );
}

#[test]
fn a_dry_budget_counts_and_cannot_run() {
    let executor = Executor::new(2);
    let plan = all(bus(1 << 20, Backing::Dry, 3, false));
    plan.budget(&executor).expect("accepted");
    let ledger = plan.bus().ledger();
    assert_eq!(ledger.total(Kind::Heap), 8 * 3503);
    assert_eq!(ledger.allocations(), 0);
    let error = plan.execute(&executor).expect_err("nothing was allocated");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

#[test]
fn a_failed_step_fails_its_readers_and_nothing_else() {
    let executor = Executor::new(4);
    let plan = all(bus(1 << 20, Backing::Real, 2, true));
    plan.budget(&executor).expect("accepted");
    let error = plan
        .execute(&executor)
        .expect_err("the failure is the run's");
    assert_eq!(error.kind, ErrorKind::Numerical);
    assert!(error.to_string().contains("the fake failure"), "{error}");
    assert!(error.to_string().contains("step 'failing'"), "{error}");
    let ran = plan.bus().shared().ran.lock().unwrap().clone();
    assert!(
        !ran.contains(&"after"),
        "the failed step's reader was skipped: {ran:?}"
    );
    for step in ["a", "b", "c", "looping", "failing"] {
        assert!(ran.contains(&step), "{step} still ran: {ran:?}");
    }
}

#[test]
fn a_step_that_reads_one_the_config_does_not_ask_for_is_refused() {
    let mut dag = Dag::new();
    dag.add(A).add(C);
    let error = dag
        .build(bus(1 << 20, Backing::Real, 1, false))
        .err()
        .expect("refused");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(
        error.message,
        "step 'c' reads step 'b', which the config does not ask for"
    );
    let mut dag = Dag::new();
    dag.add(A).add(A);
    let error = dag
        .build(bus(1 << 20, Backing::Real, 1, false))
        .err()
        .expect("refused");
    assert_eq!(error.message, "step 'a' is in the plan twice");
}

#[test]
fn the_step_graph_is_drawn() {
    let plan = all(bus(1 << 20, Backing::Real, 1, false));
    let dot = plan.dot();
    assert!(dot.contains("\"a\" -> \"b\";"));
    assert!(dot.contains("\"looping\" [shape=doublecircle];"));
}
