//! The core: steps, the slots of a bus, the declared view, and the one DAG
//! built twice on rustflow, once calling each step's `mem_budget` and once
//! its `run`.
//!
//! Nothing here names a device or a concrete step. A [`Bus`] says what is
//! shared before budget mode ([`Bus::Shared`]), what the wrapper holds for a
//! step ([`Bus::Lane`]: its streams, on a real bus), and how a step's body
//! is entered and left; `MainBus` is the real one, and the tests bring a
//! small one of their own.
//!
//! **The declaration is the edge.** A step names the steps it reads as a
//! tuple, [`Step::Reads`]. The builder makes the DAG's edges from that tuple
//! and from nothing else, and [`View::read`] compiles only for a step that
//! is in it, so a read with no edge behind it cannot be written.

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::time::{Duration, Instant};

use fb_core_error_handling::{Error, Result, Status};
use rustflow::Executor;
use rustflow::prelude::*;

use crate::budget::{Backing, Budget, Lease, Ledger};

/// Where a step runs, fixed before budget mode: stealing moves which thread
/// runs a step, never this. The only three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// On no GPU.
    Host,
    /// On gpu0, the privileged card.
    Gpu0,
    /// On every card of the run, its result reduced onto gpu0.
    AllGpus,
}

/// The shared object of a run: what is filled before budget mode, the
/// ledger, and the hooks the step wrapper calls around a body.
pub trait Bus: Send + Sync + Sized + 'static {
    /// What setup fills and every step may read without declaring it: the
    /// config, the dimensions, the device resources.
    type Shared;
    /// What the wrapper makes for a step in budget mode and hands its `run`:
    /// the step's own stream on each card it spans, on a real bus.
    type Lane: Send + 'static;
    /// Held around a `run`: the device made current, the locks of the cards
    /// whose library the step calls.
    type Hold<'a>
    where
        Self: 'a;

    /// What setup filled.
    fn shared(&self) -> &Self::Shared;
    /// The caps, counters and table.
    fn ledger(&self) -> &Ledger;
    /// Called once per step in budget mode, after its `mem_budget`: by then
    /// the ledger knows the cards whose library the step calls.
    fn make_lane(&self, step: &'static str, placement: Placement) -> Result<Self::Lane>;
    /// Called before every `run`: make the step's device current on this
    /// thread, take the lock of every card the step calls the library on,
    /// in card order, and bind those cards' handles to the step's streams.
    /// In the ledger's NaN-fill mode, fill `lease` too.
    fn acquire<'a>(
        &'a self,
        step: &'static str,
        placement: Placement,
        lane: &mut Self::Lane,
        lease: &Lease<'_>,
    ) -> Result<Self::Hold<'a>>;
    /// Called after every `run`, failed or not: wait for the step's device
    /// work on every card, so that a finished node is finished device work,
    /// and only then release what [`Bus::acquire`] took.
    fn release(&self, step: &'static str, lane: &Self::Lane, hold: Self::Hold<'_>) -> Status;
}

/// One step's place in the bus: its `Output` behind a lock used plainly,
/// and beside it the lane the wrapper keeps for it. Empty until budget mode
/// fills it.
pub struct Slot<O, L> {
    output: RwLock<Option<O>>,
    lane: Mutex<Option<L>>,
}

impl<O, L> Default for Slot<O, L> {
    fn default() -> Self {
        Slot {
            output: RwLock::new(None),
            lane: Mutex::new(None),
        }
    }
}

impl<O, L> Slot<O, L> {
    /// An empty slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether budget mode has put the step's `Output` here.
    pub fn is_filled(&self) -> bool {
        self.output
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn fill(&self, output: O, lane: L) {
        *self.output.write().unwrap_or_else(|e| e.into_inner()) = Some(output);
        *self.lane.lock().unwrap_or_else(|e| e.into_inner()) = Some(lane);
    }

    fn read(&self, step: &'static str) -> Read<'_, O> {
        let guard = self.output.read().unwrap_or_else(|e| e.into_inner());
        assert!(
            guard.is_some(),
            "step '{step}' is read before it was budgeted"
        );
        Read(guard)
    }
}

/// A reader's hold on another step's `Output`.
pub struct Read<'a, O>(RwLockReadGuard<'a, Option<O>>);

impl<O> Deref for Read<'_, O> {
    type Target = O;
    fn deref(&self) -> &O {
        self.0.as_ref().expect("checked when the guard was taken")
    }
}

/// A step: one config table, one slot, one function that counts and
/// allocates, and one that runs. A unit struct, so it is named by value:
/// `view.read(EriStep)`.
pub trait Step<B: Bus>: Copy + Send + Sync + 'static {
    /// The step's name: its row of the budget table and its node.
    const NAME: &'static str;
    /// What outlives the step's `run`: what other steps read, or what
    /// carries over a replay. `'static`, so it cannot hold a lease.
    type Output: Send + Sync + 'static;
    /// The steps whose `Output`s `run` reads: a tuple of steps, `()` for
    /// none. The DAG's edges are exactly these.
    type Reads: ReadSet<B>;

    /// The step's slot in the bus.
    fn slot(bus: &B) -> &Slot<Self::Output, B::Lane>;

    /// Where the step runs.
    fn placement(_shared: &B::Shared) -> Placement {
        Placement::Host
    }

    /// Charge what the step's `Output` holds and allocate it, declare the
    /// step's scratch and the cards whose library it calls, and return the
    /// `Output`; the wrapper puts it in the slot. Budget mode calls it once.
    /// It reads the dimensions, the step's options and the caps, never
    /// another step's `Output`.
    ///
    /// # Errors
    /// A size it cannot ask for: a limit too small for one batch.
    fn mem_budget(shared: &B::Shared, budget: &Budget<'_>) -> Result<Self::Output>;

    /// The step's work, into the buffers `mem_budget` made and the scratch
    /// it declared. It has no `Budget`, so it allocates nothing budgeted,
    /// and `scratch` is a borrow, so it keeps none of it.
    ///
    /// # Errors
    /// The step's failure: its readers are then skipped.
    fn run(
        out: &mut Self::Output,
        view: &View<'_, B, Self>,
        lane: &mut B::Lane,
        scratch: &Lease<'_>,
    ) -> Status;
}

/// A tuple of steps: what a step declares it reads.
pub trait ReadSet<B: Bus> {
    /// The names of the steps in the tuple.
    fn names() -> Vec<&'static str>;
}

impl<B: Bus> ReadSet<B> for () {
    fn names() -> Vec<&'static str> {
        Vec::new()
    }
}

/// `S` is in this tuple, at position `I`: the bound [`View::read`] puts on a
/// step's `Reads`. `I` is inferred and never written.
#[diagnostic::on_unimplemented(
    message = "this step reads a step its `Reads` does not name",
    label = "not in the reading step's `Reads`",
    note = "the DAG's edges are made from `Reads`: add this step to that tuple"
)]
pub trait Has<S, I> {}

/// A position in a tuple, for [`Has`].
pub struct At<const N: usize>;

include!("flow_tuples.rs");

/// What a step's `run` is handed: what setup filled, and the `Output`s of
/// the steps it declared. Not the bus.
pub struct View<'a, B: Bus, S: Step<B>> {
    bus: &'a B,
    step: PhantomData<S>,
}

impl<'a, B: Bus, S: Step<B>> View<'a, B, S> {
    /// What setup filled.
    pub fn shared(&self) -> &'a B::Shared {
        self.bus.shared()
    }

    /// The `Output` of `step`, which this step's `Reads` names. A step not
    /// in `Reads` does not compile (`tests/core.rs` has both cases as
    /// programs; the failing one is `experimental/rust/v0.0.2`'s doctest).
    pub fn read<T: Step<B>, I>(&self, _step: T) -> Read<'a, T::Output>
    where
        S::Reads: Has<T, I>,
    {
        T::slot(self.bus).read(T::NAME)
    }
}

type BudgetFn<B> = Arc<dyn Fn(&B) -> Status + Send + Sync>;
type RunFn<B> = Arc<dyn Fn(&B) -> Result<(Duration, Duration)> + Send + Sync>;

struct Node<B> {
    name: &'static str,
    reads: Vec<&'static str>,
    budget: BudgetFn<B>,
    run: RunFn<B>,
}

/// The steps a config asks for, before the edges are checked.
pub struct Dag<B: Bus> {
    nodes: Vec<Node<B>>,
}

impl<B: Bus> Default for Dag<B> {
    fn default() -> Self {
        Dag { nodes: Vec::new() }
    }
}

impl<B: Bus> Dag<B> {
    /// No step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `step`: its two bodies, each inside the shared wrapper, and the
    /// names it reads.
    pub fn add<S: Step<B>>(&mut self, _step: S) -> &mut Self {
        let budget: BudgetFn<B> = Arc::new(|bus: &B| {
            let budget = bus.ledger().budget(S::NAME);
            let output = S::mem_budget(bus.shared(), &budget)?;
            let lane = bus.make_lane(S::NAME, S::placement(bus.shared()))?;
            S::slot(bus).fill(output, lane);
            Ok(())
        });
        let run: RunFn<B> = Arc::new(|bus: &B| {
            let slot = S::slot(bus);
            let mut lane = slot.lane.lock().unwrap_or_else(|e| e.into_inner());
            let lane = lane
                .as_mut()
                .unwrap_or_else(|| panic!("step '{}' runs before it was budgeted", S::NAME));
            // The lease lives for this call and no longer.
            let lease = bus.ledger().lease(S::NAME);
            let hold = bus.acquire(S::NAME, S::placement(bus.shared()), lane, &lease)?;
            let mut output = slot.output.write().unwrap_or_else(|e| e.into_inner());
            let output = output.as_mut().expect("filled with the lane");
            let view = View {
                bus,
                step: PhantomData::<S>,
            };
            let start = Instant::now();
            let ran = S::run(output, &view, lane, &lease);
            let run = start.elapsed();
            // The wrapper's sync, then the locks: a finished node is
            // finished device work, whether `run` failed or not.
            let start = Instant::now();
            let left = bus.release(S::NAME, lane, hold);
            ran?;
            left?;
            Ok((run, start.elapsed()))
        });
        self.nodes.push(Node {
            name: S::NAME,
            reads: S::Reads::names(),
            budget,
            run,
        });
        self
    }

    /// Check the steps close and make the plan over `bus`.
    ///
    /// # Errors
    /// `InvalidConfig` for a step added twice, a step that reads itself, a
    /// step that reads one the config does not ask for (naming both), or a
    /// cycle.
    pub fn build(self, bus: Arc<B>) -> Result<Plan<B>> {
        let names: Vec<&'static str> = self.nodes.iter().map(|n| n.name).collect();
        for (i, name) in names.iter().enumerate() {
            if names[..i].contains(name) {
                return Err(Error::config(format!("step '{name}' is in the plan twice")));
            }
        }
        let mut preds = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let mut of_node = Vec::with_capacity(node.reads.len());
            for read in &node.reads {
                if *read == node.name {
                    return Err(Error::config(format!("step '{read}' reads itself")));
                }
                match names.iter().position(|n| n == read) {
                    Some(p) => of_node.push(p),
                    None => {
                        return Err(Error::config(format!(
                            "step '{}' reads step '{read}', which the config does not ask for",
                            node.name
                        )));
                    }
                }
            }
            of_node.sort_unstable();
            of_node.dedup();
            preds.push(of_node);
        }
        // Kahn: every step is reachable from the sources.
        let mut left: Vec<usize> = preds.iter().map(Vec::len).collect();
        let mut ready: Vec<usize> = (0..left.len()).filter(|&i| left[i] == 0).collect();
        let mut seen = 0;
        while let Some(done) = ready.pop() {
            seen += 1;
            for (i, of_node) in preds.iter().enumerate() {
                if of_node.contains(&done) {
                    left[i] -= 1;
                    if left[i] == 0 {
                        ready.push(i);
                    }
                }
            }
        }
        if seen != names.len() {
            let stuck: Vec<&str> = (0..left.len())
                .filter(|&i| left[i] > 0)
                .map(|i| names[i])
                .collect();
            return Err(Error::config(format!(
                "the steps' reads make a cycle through {}",
                stuck.join(", ")
            )));
        }
        Ok(Plan {
            bus,
            nodes: Arc::new(self.nodes),
            preds: Arc::new(preds),
        })
    }
}

/// One step's time in an execute.
#[derive(Clone, Debug)]
pub struct StepTime {
    /// The step.
    pub name: &'static str,
    /// The time inside `run`.
    pub run: Duration,
    /// The time inside the wrapper's sync and release.
    pub sync: Duration,
    /// When the node began (before its locks), from the start of the
    /// execute.
    pub began: Duration,
    /// When the wrapper's release returned.
    pub ended: Duration,
}

/// What one mode's run of the graph leaves behind.
struct RunState {
    start: Instant,
    failed: Vec<AtomicBool>,
    error: Mutex<Option<Error>>,
    times: Mutex<Vec<StepTime>>,
}

impl RunState {
    fn new(names: impl Iterator<Item = &'static str>) -> Arc<RunState> {
        let times: Vec<StepTime> = names
            .map(|name| StepTime {
                name,
                run: Duration::ZERO,
                sync: Duration::ZERO,
                began: Duration::ZERO,
                ended: Duration::ZERO,
            })
            .collect();
        Arc::new(RunState {
            start: Instant::now(),
            failed: (0..times.len()).map(|_| AtomicBool::new(false)).collect(),
            error: Mutex::new(None),
            times: Mutex::new(times),
        })
    }

    fn fail(&self, i: usize, error: Error) {
        self.failed[i].store(true, Ordering::Release);
        self.error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert(error);
    }

    fn take_error(&self) -> Option<Error> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// The checked step graph over a bus: built into a rustflow `Taskflow`
/// once per mode.
pub struct Plan<B: Bus> {
    bus: Arc<B>,
    nodes: Arc<Vec<Node<B>>>,
    preds: Arc<Vec<Vec<usize>>>,
}

impl<B: Bus> Plan<B> {
    /// The bus.
    pub fn bus(&self) -> &Arc<B> {
        &self.bus
    }

    /// The `Output` of `step`, for whoever holds the plan after a run.
    pub fn output<S: Step<B>>(&self, _step: S) -> Read<'_, S::Output> {
        S::slot(&self.bus).read(S::NAME)
    }

    /// The steps' names, in the order they were added.
    pub fn names(&self) -> Vec<&'static str> {
        self.nodes.iter().map(|n| n.name).collect()
    }

    /// For each step, the indices of the steps it reads.
    pub fn preds(&self) -> &[Vec<usize>] {
        &self.preds
    }

    /// The steps and, for each, the steps it reads: the DAG's edges.
    pub fn edges(&self) -> Vec<(&'static str, Vec<&'static str>)> {
        self.nodes
            .iter()
            .zip(self.preds.iter())
            .map(|(n, p)| (n.name, p.iter().map(|&i| self.nodes[i].name).collect()))
            .collect()
    }

    /// Budget mode: every step's `mem_budget`, over the same edges, then the
    /// final node, [`Ledger::finish`]: the closure, the scratch packing and
    /// the handle workspace, the host check, accept or refuse. The totals
    /// are order-independent, so accept or refuse is the same every run,
    /// and so are the offsets and the table.
    ///
    /// # Errors
    /// A step's `mem_budget` error; or the refusal, carrying the table.
    pub fn budget(&self, executor: &Executor) -> Status {
        let state = RunState::new(self.nodes.iter().map(|n| n.name));
        let mut flow = Taskflow::with_name("budget");
        let mut tasks = Vec::with_capacity(self.nodes.len());
        for i in 0..self.nodes.len() {
            let (bus, nodes, state) = (
                Arc::clone(&self.bus),
                Arc::clone(&self.nodes),
                Arc::clone(&state),
            );
            let task = flow.emplace(move || {
                if let Err(e) = (nodes[i].budget)(&bus) {
                    let name = nodes[i].name;
                    state.fail(i, e.with_context(format!("step '{name}', mem_budget")));
                }
            });
            flow.task(task).name(self.nodes[i].name);
            tasks.push(task);
        }
        for (i, of_node) in self.preds.iter().enumerate() {
            for &p in of_node {
                flow.task(tasks[p]).precede([tasks[i]]);
            }
        }
        let (bus, nodes, preds, final_state) = (
            Arc::clone(&self.bus),
            Arc::clone(&self.nodes),
            Arc::clone(&self.preds),
            Arc::clone(&state),
        );
        let finish = flow.emplace(move || {
            if final_state.failed.iter().any(|f| f.load(Ordering::Acquire)) {
                return;
            }
            let names: Vec<&'static str> = nodes.iter().map(|n| n.name).collect();
            if let Err(e) = bus.ledger().finish(&names, &preds) {
                final_state
                    .error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert(e);
            }
        });
        flow.task(finish).name("budget: pack, accept or refuse");
        flow.task(finish).succeed(tasks.iter().copied());
        wait(executor, &flow)?;
        state.take_error().map_or(Ok(()), Err)
    }

    /// Execute mode: every step's `run` inside the wrapper, a step with a
    /// failed predecessor skipped. Allocates nothing budgeted. May be called
    /// again: a replay writes into the same buffers and the same scratch
    /// regions.
    ///
    /// # Errors
    /// `InvalidConfig` if the plan was not budgeted and accepted, with
    /// bytes behind it; otherwise the first failing step's error.
    pub fn execute(&self, executor: &Executor) -> Result<Vec<StepTime>> {
        let ledger = self.bus.ledger();
        if !ledger.sealed() || ledger.over() || ledger.backing() == Backing::Dry {
            return Err(Error::config(
                "execute: the plan was not budgeted and accepted with real allocations",
            ));
        }
        let state = RunState::new(self.nodes.iter().map(|n| n.name));
        let mut flow = Taskflow::with_name("execute");
        let mut tasks = Vec::with_capacity(self.nodes.len());
        for i in 0..self.nodes.len() {
            let (bus, nodes, preds, state) = (
                Arc::clone(&self.bus),
                Arc::clone(&self.nodes),
                Arc::clone(&self.preds),
                Arc::clone(&state),
            );
            let task = flow.emplace(move || {
                if preds[i]
                    .iter()
                    .any(|&p| state.failed[p].load(Ordering::Acquire))
                {
                    state.failed[i].store(true, Ordering::Release);
                    log::warn!("step '{}' skipped: a step it reads failed", nodes[i].name);
                    return;
                }
                let began = state.start.elapsed();
                match (nodes[i].run)(&bus) {
                    Ok((run, sync)) => {
                        let mut times = state.times.lock().unwrap_or_else(|e| e.into_inner());
                        times[i].began = began;
                        times[i].ended = state.start.elapsed();
                        times[i].run = run;
                        times[i].sync = sync;
                    }
                    Err(e) => {
                        let name = nodes[i].name;
                        state.fail(i, e.with_context(format!("step '{name}'")));
                    }
                }
            });
            flow.task(task).name(self.nodes[i].name);
            tasks.push(task);
        }
        for (i, of_node) in self.preds.iter().enumerate() {
            for &p in of_node {
                flow.task(tasks[p]).precede([tasks[i]]);
            }
        }
        wait(executor, &flow)?;
        if let Some(error) = state.take_error() {
            return Err(error);
        }
        let times = state
            .times
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Ok(times)
    }
}

/// Run `flow` and wait: a task's panic is this run's error.
fn wait(executor: &Executor, flow: &Taskflow) -> Status {
    executor.run(flow).wait().map_err(|e| match e {
        RunError::Panicked(payload) => {
            let what = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("a payload that is not a string");
            Error::config(format!("a task panicked: {what}"))
        }
        _ => Error::config("the run failed"),
    })
}
