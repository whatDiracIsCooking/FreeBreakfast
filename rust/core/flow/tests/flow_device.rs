//! The step wrapper on a card (tier 4): a real plan over device 0, where
//! the wrapper makes the step's stream, takes its declared locks in rank
//! order, binds gpu0's handle workspace, fills its device scratch with NaN
//! in the ledger's test mode, and syncs before it releases.

use std::sync::Arc;

use fb_core_budget::{Backing, Budget, Caps, Kind, Layout, Lease, Ledger, LockId, ManagerId, Span};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Bus, Dag, Executor, Held, Lane, Locks, Placement, Slot, Step, View};

const N: usize = 1024;

struct Card {
    ledger: Ledger,
    locks: Locks,
    x: Slot<Out>,
}

#[derive(Default)]
struct Out {
    /// The step's device scratch as each `run` found it.
    found: Vec<f64>,
    /// The locks `run` was handed, as the wrapper took them.
    taken: Vec<LockId>,
}

impl Bus for Card {
    type Shared = ();
    fn shared(&self) -> &() {
        &()
    }
    fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    fn locks(&self) -> &Locks {
        &self.locks
    }
}

fn layout() -> (Layout, Span<f64>, Span<f64>) {
    let mut l = Layout::new();
    let device = l.span::<f64>(N);
    let staged = l.span::<f64>(N);
    (l, device, staged)
}

#[derive(Clone, Copy)]
struct X;
impl Step<Card> for X {
    const NAME: &'static str = "x";
    type Output = Out;
    type Reads = ();
    fn slot(bus: &Card) -> &Slot<Out> {
        &bus.x
    }
    fn placement(_: &()) -> Placement {
        Placement::Gpu0
    }
    fn mem_budget(_: &(), budget: &Budget<'_>) -> Result<Out> {
        let bytes = layout().0.bytes();
        budget.scratch(0, Kind::Device, bytes);
        budget.scratch(0, Kind::Pinned, bytes);
        // Declared highest first: the wrapper takes them by rank.
        budget.locks([LockId::Gpu(0), LockId::Manager(ManagerId::Eri)]);
        Ok(Out::default())
    }
    fn run(
        out: &mut Out,
        _: &View<'_, Card, Self>,
        held: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        out.taken = held.taken().to_vec();
        let (_, device, staged) = layout();
        let mut device = scratch.device(0, device);
        let host = scratch.pinned(0, staged);
        let stream = &lane.gpus()[0].streams()[0];
        // A test reading back: the token's sync, before the wrapper's.
        let host = stream.download(host, &device)?.sync()?;
        out.found.push(host[0]);
        out.found.push(host[N - 1]);
        // Something of its own for the next run to find under the fill,
        // on gpu0's stream, which the wrapper syncs too.
        held.gpu(0).stream().fill(&mut device, 0)?;
        Ok(())
    }
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_wrapper_takes_the_locks_by_rank_and_fills_device_scratch_with_nan() {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let bus = Arc::new(Card {
        ledger: Ledger::new(caps, Backing::Real, vec![0], true),
        locks: Locks::new(&[0]),
        x: Slot::new(),
    });
    let mut dag = Dag::new();
    dag.add(X);
    let plan = dag.build(bus).unwrap();
    let executor = Executor::new(2);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();
    plan.execute(&executor).unwrap();
    assert!(
        plan.bus().ledger.take_workspace(0).is_none(),
        "the wrapper bound gpu0's workspace"
    );
    let x = plan.output(X);
    assert_eq!(x.taken, [LockId::Manager(ManagerId::Eri), LockId::Gpu(0)]);
    // NaN on entry both runs, though the first left zeros behind.
    assert_eq!(x.found.len(), 4);
    assert!(x.found.iter().all(|v| v.is_nan()), "{:?}", x.found);
}
