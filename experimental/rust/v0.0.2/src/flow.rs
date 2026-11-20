//! The core: steps, the slots of a bus, the declared view, and the one DAG
//! built twice on rustflow, once calling each step's `mem_budget` and once
//! its `run`.
//!
//! Nothing here names a device or a concrete step. A [`Bus`] says what is
//! shared before budget mode ([`Bus::Shared`]), what the wrapper holds for a
//! step while it runs ([`Bus::Lane`], a stream on a real bus), and how a
//! step's body is entered and left; `MainBus` is the real one, and the tests
//! bring a small one of their own.
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

use crate::budget::{Backing, Budget, Ledger};

/// Where a step's memory lives, fixed before budget mode: stealing moves
/// which thread runs a step, never this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// No device.
    Host,
    /// One device; a one-GPU step's is gpu0, the privileged card.
    Gpu(usize),
    /// Every device of the run, its result reduced onto gpu0.
    AllGpus,
}

/// What a step's `run` says of itself when it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// The step is finished; its readers may run.
    Done,
    /// Run it again: one more trip of a [`Step::LOOPS`] step's loop.
    Again,
}

/// The shared object of a run: what is filled before budget mode, the
/// ledger, and the hooks the step wrapper calls around a body.
pub trait Bus: Send + Sync + Sized + 'static {
    /// What setup fills and every step may read without declaring it: the
    /// config, the dimensions, the device resources.
    type Shared;
    /// What the wrapper makes for a step in budget mode and hands its `run`:
    /// the step's own stream, on a real bus.
    type Lane: Send + Sync + 'static;
    /// Held around a body: the device made current.
    type Guard;

    /// What setup filled.
    fn shared(&self) -> &Self::Shared;
    /// The caps, counters and table.
    fn ledger(&self) -> &Ledger;
    /// Called before a body, `mem_budget` or `run`: make the step's device
    /// current on this thread, whichever worker it is.
    fn enter(&self, placement: Placement) -> Self::Guard;
    /// Called once per step in budget mode, inside [`Bus::enter`].
    fn make_lane(&self, placement: Placement) -> Result<Self::Lane>;
    /// Called after every `run`: wait for the step's device work, so that a
    /// finished node is finished device work.
    fn leave(&self, lane: &Self::Lane) -> Status;
    /// Called by the budget DAG's final node for each device with a
    /// workspace: allocate `bytes` and hand them to the library handles.
    fn bind_workspace(&self, device: usize, bytes: u64) -> Status;
}

/// One step's place in the bus: its `Output` and its lane, behind a lock
/// used plainly. Empty until budget mode fills it.
pub struct Slot<O, L>(RwLock<Option<(O, L)>>);

impl<O, L> Default for Slot<O, L> {
    fn default() -> Self {
        Slot(RwLock::new(None))
    }
}

impl<O, L> Slot<O, L> {
    /// An empty slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether budget mode has put the step's `Output` here.
    pub fn is_filled(&self) -> bool {
        self.0.read().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    fn fill(&self, output: O, lane: L) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = Some((output, lane));
    }

    fn read(&self, step: &'static str) -> Read<'_, O, L> {
        let guard = self.0.read().unwrap_or_else(|e| e.into_inner());
        assert!(
            guard.is_some(),
            "step '{step}' is read before it was budgeted"
        );
        Read(guard)
    }
}

/// A reader's hold on another step's `Output`.
pub struct Read<'a, O, L>(RwLockReadGuard<'a, Option<(O, L)>>);

impl<O, L> Deref for Read<'_, O, L> {
    type Target = O;
    fn deref(&self) -> &O {
        &self.0.as_ref().expect("checked when the guard was taken").0
    }
}

/// A step: one config table, one slot, one function that counts and
/// allocates, and one that runs. A unit struct, so it is named by value:
/// `view.read(OverlapStep)`.
pub trait Step<B: Bus>: Copy + Send + Sync + 'static {
    /// The step's name: its row of the budget table and its node.
    const NAME: &'static str;
    /// Whether `run` may return [`Next::Again`]: the builder then makes the
    /// step a condition loop of the graph, each trip one `run` and one sync.
    const LOOPS: bool = false;
    /// Everything the step allocates: the products other steps read, and
    /// its private buffers.
    type Output: Send + Sync + 'static;
    /// The steps whose `Output`s `run` reads: a tuple of steps, `()` for
    /// none. The DAG's edges are exactly these.
    type Reads: ReadSet<B>;

    /// The step's slot in the bus.
    fn slot(bus: &B) -> &Slot<Self::Output, B::Lane>;

    /// Where the step's memory lives.
    fn placement(_shared: &B::Shared) -> Placement {
        Placement::Host
    }

    /// Charge what the step allocates, allocate it, and return the
    /// `Output`; the wrapper puts it in the slot. Budget mode calls it once.
    /// It reads the dimensions and the step's options, never another step's
    /// `Output`.
    ///
    /// # Errors
    /// A size it cannot ask for (a failed workspace query).
    fn mem_budget(shared: &B::Shared, lane: &B::Lane, budget: &Budget<'_>) -> Result<Self::Output>;

    /// The step's work, into the buffers `mem_budget` made. It has no
    /// `Budget`, so it allocates nothing budgeted.
    ///
    /// # Errors
    /// The step's failure: its readers are then skipped.
    fn run(out: &mut Self::Output, view: &View<'_, B, Self>) -> Result<Next>;
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
// The message cannot name the step: with the position inferred, the compiler
// knows `S` only as `_` when the bound fails. The label lands on the argument.
#[diagnostic::on_unimplemented(
    message = "this step reads a step its `Reads` does not name",
    label = "not in the reading step's `Reads`",
    note = "the DAG's edges are made from `Reads`: add this step to that tuple"
)]
pub trait Has<S, I> {}

/// A position in a tuple, for [`Has`].
pub struct At<const N: usize>;

include!("flow_tuples.rs");

/// What a step's `run` is handed: its lane, what setup filled, and the
/// `Output`s of the steps it declared. Not the bus.
pub struct View<'a, B: Bus, S: Step<B>> {
    bus: &'a B,
    lane: &'a B::Lane,
    step: PhantomData<S>,
}

impl<'a, B: Bus, S: Step<B>> View<'a, B, S> {
    /// What setup filled.
    pub fn shared(&self) -> &'a B::Shared {
        self.bus.shared()
    }

    /// The step's own lane.
    pub fn lane(&self) -> &'a B::Lane {
        self.lane
    }

    /// The `Output` of `step`, which this step's `Reads` names. A step not
    /// in `Reads` does not compile.
    ///
    /// A step that declares what it reads:
    ///
    /// ```no_run
    #[doc = include_str!("../tests/doc/bus.rs")]
    /// #[derive(Clone, Copy)]
    /// struct Reader;
    /// impl Step<TestBus> for Reader {
    ///     const NAME: &'static str = "reader";
    ///     type Output = f64;
    ///     type Reads = (Source,);
    ///     fn slot(bus: &TestBus) -> &Slot<f64, ()> { &bus.reader }
    ///     fn mem_budget(_: &(), _: &(), _: &Budget<'_>) -> Result<f64> { Ok(0.0) }
    ///     fn run(out: &mut f64, view: &View<'_, TestBus, Self>) -> Result<Next> {
    ///         *out = *view.read(Source);
    ///         Ok(Next::Done)
    ///     }
    /// }
    /// ```
    ///
    /// The same step with the declaration left out is not a program: the
    /// read has no edge behind it.
    ///
    /// ```compile_fail,E0277
    #[doc = include_str!("../tests/doc/bus.rs")]
    /// #[derive(Clone, Copy)]
    /// struct Reader;
    /// impl Step<TestBus> for Reader {
    ///     const NAME: &'static str = "reader";
    ///     type Output = f64;
    ///     type Reads = ();
    ///     fn slot(bus: &TestBus) -> &Slot<f64, ()> { &bus.reader }
    ///     fn mem_budget(_: &(), _: &(), _: &Budget<'_>) -> Result<f64> { Ok(0.0) }
    ///     fn run(out: &mut f64, view: &View<'_, TestBus, Self>) -> Result<Next> {
    ///         *out = *view.read(Source);
    ///         Ok(Next::Done)
    ///     }
    /// }
    /// ```
    pub fn read<T: Step<B>, I>(&self, _step: T) -> Read<'a, T::Output, B::Lane>
    where
        S::Reads: Has<T, I>,
    {
        T::slot(self.bus).read(T::NAME)
    }
}

/// One trip of a step's `run`, timed.
struct Trip {
    next: Next,
    run: Duration,
    sync: Duration,
}

type BudgetFn<B> = Arc<dyn Fn(&B) -> Status + Send + Sync>;
type RunFn<B> = Arc<dyn Fn(&B) -> Result<Trip> + Send + Sync>;

struct Node<B> {
    name: &'static str,
    reads: Vec<&'static str>,
    loops: bool,
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
            let placement = S::placement(bus.shared());
            let _current = bus.enter(placement);
            let lane = bus.make_lane(placement)?;
            let budget = bus.ledger().budget(S::NAME);
            let output = S::mem_budget(bus.shared(), &lane, &budget)?;
            S::slot(bus).fill(output, lane);
            Ok(())
        });
        let run: RunFn<B> = Arc::new(|bus: &B| {
            let _current = bus.enter(S::placement(bus.shared()));
            let mut state = S::slot(bus).0.write().unwrap_or_else(|e| e.into_inner());
            let (output, lane) = state
                .as_mut()
                .unwrap_or_else(|| panic!("step '{}' runs before it was budgeted", S::NAME));
            let lane: &B::Lane = lane;
            let view = View {
                bus,
                lane,
                step: PhantomData::<S>,
            };
            let start = Instant::now();
            let next = S::run(output, &view)?;
            let run = start.elapsed();
            // The wrapper's sync: a finished node is finished device work.
            let start = Instant::now();
            bus.leave(lane)?;
            Ok(Trip {
                next,
                run,
                sync: start.elapsed(),
            })
        });
        self.nodes.push(Node {
            name: S::NAME,
            reads: S::Reads::names(),
            loops: S::LOOPS,
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

/// One step's time in an execute: every trip of its `run`, and of the
/// wrapper's sync after it.
#[derive(Clone, Debug)]
pub struct StepTime {
    /// The step.
    pub name: &'static str,
    /// How many times `run` ran: 1, or a looping step's trips.
    pub trips: u32,
    /// The time inside `run`.
    pub run: Duration,
    /// The time inside the wrapper's sync.
    pub sync: Duration,
    /// When the first trip began, from the start of the execute.
    pub began: Duration,
    /// When the last trip's sync returned, from the start of the execute.
    pub ended: Duration,
}

/// What one mode's run of the graph leaves behind.
struct RunState {
    start: Instant,
    failed: Vec<AtomicBool>,
    again: Vec<AtomicBool>,
    error: Mutex<Option<Error>>,
    times: Mutex<Vec<StepTime>>,
}

impl RunState {
    fn new(names: impl Iterator<Item = &'static str>) -> Arc<RunState> {
        let times: Vec<StepTime> = names
            .map(|name| StepTime {
                name,
                trips: 0,
                run: Duration::ZERO,
                sync: Duration::ZERO,
                began: Duration::ZERO,
                ended: Duration::ZERO,
            })
            .collect();
        let flags = || (0..times.len()).map(|_| AtomicBool::new(false)).collect();
        Arc::new(RunState {
            start: Instant::now(),
            failed: flags(),
            again: flags(),
            error: Mutex::new(None),
            times: Mutex::new(times),
        })
    }

    fn fail(&self, i: usize, error: Error) {
        self.failed[i].store(true, Ordering::Release);
        self.again[i].store(false, Ordering::Release);
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
    pub fn output<S: Step<B>>(&self, _step: S) -> Read<'_, S::Output, B::Lane> {
        S::slot(&self.bus).read(S::NAME)
    }

    /// The steps and, for each, the steps it reads: the DAG's edges.
    pub fn edges(&self) -> Vec<(&'static str, Vec<&'static str>)> {
        self.nodes
            .iter()
            .zip(self.preds.iter())
            .map(|(n, p)| (n.name, p.iter().map(|&i| self.nodes[i].name).collect()))
            .collect()
    }

    /// The step graph as Graphviz DOT; a looping step is drawn doubled.
    pub fn dot(&self) -> String {
        let mut out = String::from("digraph steps {\n  rankdir=TB;\n");
        for node in self.nodes.iter() {
            let shape = if node.loops { "doublecircle" } else { "box" };
            out.push_str(&format!("  \"{}\" [shape={shape}];\n", node.name));
        }
        for (name, reads) in self.edges() {
            for read in reads {
                out.push_str(&format!("  \"{read}\" -> \"{name}\";\n"));
            }
        }
        out.push_str("}\n");
        out
    }

    /// Budget mode: every step's `mem_budget`, over the same edges, then the
    /// final node, [`Ledger::finish`]. The total is order-independent, so
    /// accept or refuse is the same every run, and so is the table.
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
        let last = self.nodes.len();
        let (bus, final_state) = (Arc::clone(&self.bus), Arc::clone(&state));
        let finish = flow.emplace(move || {
            if final_state.failed.iter().any(|f| f.load(Ordering::Acquire)) {
                return;
            }
            let bound = bus
                .ledger()
                .finish(|device, bytes| bus.bind_workspace(device, bytes));
            if let Err(e) = bound {
                final_state
                    .error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert(e);
            }
        });
        flow.task(finish).name("budget: accept or refuse");
        flow.task(finish).succeed(tasks.iter().copied());
        debug_assert_eq!(flow.num_tasks(), last + 1);
        wait(executor, &flow)?;
        state.take_error().map_or(Ok(()), Err)
    }

    /// Execute mode: every step's `run` inside the wrapper, a step with a
    /// failed predecessor skipped. Allocates nothing budgeted. May be called
    /// again: a replay writes into the same buffers.
    ///
    /// # Errors
    /// `InvalidConfig` if the plan was not budgeted and accepted, with real
    /// allocations; otherwise the first failing step's error.
    pub fn execute(&self, executor: &Executor) -> Result<Vec<StepTime>> {
        let ledger = self.bus.ledger();
        if !ledger.sealed() || ledger.over() || ledger.backing() != Backing::Real {
            return Err(Error::config(
                "execute: the plan was not budgeted and accepted with real allocations",
            ));
        }
        let state = RunState::new(self.nodes.iter().map(|n| n.name));
        let mut flow = Taskflow::with_name("execute");
        // A step is one task, or, when it loops, an entry, the body, a
        // condition and an exit: a condition's edge is weak, so the body
        // needs a strong predecessor of its own to be a source at all.
        let mut entry = Vec::with_capacity(self.nodes.len());
        let mut exit = Vec::with_capacity(self.nodes.len());
        for i in 0..self.nodes.len() {
            let (bus, nodes, preds, run_state) = (
                Arc::clone(&self.bus),
                Arc::clone(&self.nodes),
                Arc::clone(&self.preds),
                Arc::clone(&state),
            );
            let body = flow.emplace(move || {
                let state = &run_state;
                let blocked = state.failed[i].load(Ordering::Acquire)
                    || preds[i]
                        .iter()
                        .any(|&p| state.failed[p].load(Ordering::Acquire));
                if blocked {
                    state.failed[i].store(true, Ordering::Release);
                    state.again[i].store(false, Ordering::Release);
                    log::warn!("step '{}' skipped: a step it reads failed", nodes[i].name);
                    return;
                }
                let began = state.start.elapsed();
                match (nodes[i].run)(&bus) {
                    Ok(trip) => {
                        state.again[i].store(trip.next == Next::Again, Ordering::Release);
                        let mut times = state.times.lock().unwrap_or_else(|e| e.into_inner());
                        if times[i].trips == 0 {
                            times[i].began = began;
                        }
                        times[i].ended = state.start.elapsed();
                        times[i].trips += 1;
                        times[i].run += trip.run;
                        times[i].sync += trip.sync;
                    }
                    Err(e) => {
                        let name = nodes[i].name;
                        state.fail(i, e.with_context(format!("step '{name}'")));
                    }
                }
            });
            let name = self.nodes[i].name;
            flow.task(body).name(name);
            if self.nodes[i].loops {
                let enter = flow.emplace(|| {});
                let again_state = Arc::clone(&state);
                let again = flow.emplace_condition(move || {
                    i32::from(!again_state.again[i].load(Ordering::Acquire))
                });
                let done = flow.emplace(|| {});
                flow.task(enter)
                    .name(format!("{name}: enter"))
                    .precede([body]);
                flow.task(body).precede([again]);
                // 0 loops back to the body, 1 goes on.
                flow.task(again)
                    .name(format!("{name}: again?"))
                    .precede([body, done]);
                flow.task(done).name(format!("{name}: done"));
                entry.push(enter);
                exit.push(done);
            } else {
                entry.push(body);
                exit.push(body);
            }
        }
        for (i, of_node) in self.preds.iter().enumerate() {
            for &p in of_node {
                flow.task(exit[p]).precede([entry[i]]);
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
