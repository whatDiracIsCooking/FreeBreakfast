//! The steps a config asks for ([`Dag`]), checked to close, and the plan
//! over them ([`Plan`]): one graph, built into a rustflow `Taskflow` once
//! per mode.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use fb_core_budget::{AllocationRecord, Backing, RECORD_THRESHOLD, bytes};
use fb_core_error_handling::{Error, Result, Status};
use rustflow::prelude::{Executor, FlowBuilder, RunError, Task, Taskflow};

use crate::wrapper::{acquire, make_lane, release};
use crate::{Bus, Read, ReadSet, Step, View, locked};

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

/// Budget mode's body of `S`: its `mem_budget`, then its lane, both into
/// its slot.
fn budget_body<B: Bus, S: Step<B>>(bus: &B) -> Status {
    let budget = bus.ledger().budget(S::NAME);
    let output = S::mem_budget(bus.shared(), &budget)?;
    let lane = make_lane(bus, S::NAME, S::placement(bus.shared()), S::STREAMS)?;
    S::slot(bus).fill(output, lane);
    Ok(())
}

/// Execute mode's body of `S`: its `run` inside the wrapper, and the time
/// inside `run` and inside the wrapper's sync and release.
///
/// # Panics
/// If the step was not budgeted, which [`Plan::execute`]'s check of an
/// accepted budget rules out.
fn run_body<B: Bus, S: Step<B>>(bus: &B) -> Result<(Duration, Duration)> {
    let slot = S::slot(bus);
    let lane = locked(&slot.lane);
    let lane = lane
        .as_ref()
        .unwrap_or_else(|| panic!("step '{}' runs before it was budgeted", S::NAME));
    // The lease lives for this call and no longer.
    let mut lease = bus.ledger().lease(S::NAME);
    let placement = S::placement(bus.shared());
    let mut held = acquire(bus, S::NAME, placement, lane, &mut lease)?;
    let mut output = slot.output.write().unwrap_or_else(PoisonError::into_inner);
    let output = output
        .as_mut()
        .unwrap_or_else(|| panic!("step '{}' runs before it was budgeted", S::NAME));
    let view = View {
        bus,
        step: PhantomData::<S>,
    };
    let start = Instant::now();
    // Watched by the allocation record, when the binary installs it (the
    // test tiers): one allocation past its threshold on the plain heap is
    // large host data, which belongs in a `HeapSlab`.
    let (ran, largest) = AllocationRecord::watch(|| S::run(output, &view, &mut held, lane, &lease));
    let run = start.elapsed();
    // The wrapper's sync, then the locks: a finished node is finished
    // device work, whether `run` failed or not.
    let start = Instant::now();
    let left = release(lane, held);
    ran?;
    left?;
    if let Some(largest) = largest {
        return Err(Error::config(format!(
            "step '{}' made one allocation of {} on the plain heap in its run, past the \
             allocation record's {}: large host data is a HeapSlab its mem_budget makes \
             (docs/memory.md, \"The budget\")",
            S::NAME,
            bytes(largest as u64),
            bytes(RECORD_THRESHOLD as u64)
        )));
    }
    Ok((run, start.elapsed()))
}

impl<B: Bus> Dag<B> {
    /// No step.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `step`: its two bodies, each inside the wrapper, and the names
    /// it reads.
    pub fn add<S: Step<B>>(&mut self, _step: S) -> &mut Self {
        self.nodes.push(Node {
            name: S::NAME,
            reads: S::Reads::names(),
            budget: Arc::new(budget_body::<B, S>),
            run: Arc::new(run_body::<B, S>),
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
        // Kahn's, the lowest ready index first: one topological order,
        // the same every run, which ranks the first error.
        let mut left: Vec<usize> = preds.iter().map(Vec::len).collect();
        let mut rank = vec![usize::MAX; names.len()];
        let mut ready: Vec<usize> = (0..left.len()).filter(|&i| left[i] == 0).collect();
        let mut seen = 0;
        while let Some(at) = ready
            .iter()
            .enumerate()
            .min_by_key(|&(_, &i)| i)
            .map(|(at, _)| at)
        {
            let done = ready.swap_remove(at);
            rank[done] = seen;
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
            rank,
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

/// What one mode's run of the graph leaves behind: a failed flag per node,
/// each node's error, and the times.
struct RunState {
    start: Instant,
    failed: Vec<AtomicBool>,
    errors: Mutex<Vec<(usize, Error)>>,
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
            errors: Mutex::new(Vec::new()),
            times: Mutex::new(times),
        })
    }

    fn fail(&self, i: usize, error: Error) {
        self.failed[i].store(true, Ordering::Release);
        locked(&self.errors).push((i, error));
    }

    /// The error of the first failed node in topological order, the others
    /// on its context in that order: the same report at every worker count
    /// (`docs/diagnostics.md`, "One worker fails").
    fn take_error(&self, rank: &[usize]) -> Option<Error> {
        let mut errors = std::mem::take(&mut *locked(&self.errors));
        errors.sort_by_key(|&(i, _)| rank.get(i).copied().unwrap_or(usize::MAX));
        let mut errors = errors.into_iter().map(|(_, e)| e);
        let first = errors.next()?;
        Some(errors.fold(first, |first, also| {
            first.with_context(format!("and also: {also}"))
        }))
    }
}

/// The checked step graph over a bus: built into a rustflow `Taskflow`
/// once per mode.
pub struct Plan<B: Bus> {
    bus: Arc<B>,
    nodes: Arc<Vec<Node<B>>>,
    preds: Arc<Vec<Vec<usize>>>,
    /// Each step's place in the one topological order.
    rank: Vec<usize>,
}

impl<B: Bus> Plan<B> {
    /// The bus.
    #[must_use]
    pub fn bus(&self) -> &Arc<B> {
        &self.bus
    }

    /// The `Output` of `step`, for whoever holds the plan after a run.
    ///
    /// # Panics
    /// If budget mode has not filled `step`'s slot.
    pub fn output<S: Step<B>>(&self, _step: S) -> Read<'_, S::Output> {
        S::slot(&self.bus).read(S::NAME)
    }

    /// The steps and, for each, the steps it reads: the DAG's edges.
    #[must_use]
    pub fn edges(&self) -> Vec<(&'static str, Vec<&'static str>)> {
        self.nodes
            .iter()
            .zip(self.preds.iter())
            .map(|(n, p)| (n.name, p.iter().map(|&i| self.nodes[i].name).collect()))
            .collect()
    }

    /// One task per step, its body `body(i)`, each after the steps it reads.
    fn flow(
        &self,
        name: &str,
        body: impl Fn(usize) -> Box<dyn FnMut() + Send>,
    ) -> (Taskflow, Vec<Task>) {
        let mut flow = Taskflow::with_name(name);
        let mut tasks = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let task = flow.emplace(body(i));
            flow.task(task).name(node.name);
            tasks.push(task);
        }
        for (i, of_node) in self.preds.iter().enumerate() {
            for &p in of_node {
                flow.task(tasks[p]).precede([tasks[i]]);
            }
        }
        (flow, tasks)
    }

    /// Budget mode: every step's `mem_budget` over the same edges, then
    /// the final node, [`Ledger::finish`](fb_core_budget::Ledger::finish):
    /// the closure, the scratch packing and the handle workspace, accept
    /// or refuse. The totals are order-independent, so accept or refuse is
    /// the same every run, and so are the offsets and the table.
    ///
    /// # Errors
    /// The first failing `mem_budget`'s error in topological order; or the
    /// refusal, carrying the table.
    pub fn budget(&self, executor: &Executor) -> Status {
        let state = RunState::new(self.nodes.iter().map(|n| n.name));
        let (mut flow, tasks) = self.flow("budget", |i| {
            let (bus, nodes, state) = (
                Arc::clone(&self.bus),
                Arc::clone(&self.nodes),
                Arc::clone(&state),
            );
            Box::new(move || {
                if let Err(e) = (nodes[i].budget)(&bus) {
                    let name = nodes[i].name;
                    state.fail(i, e.with_context(format!("step '{name}', mem_budget")));
                }
            })
        });
        let (bus, nodes, preds, last) = (
            Arc::clone(&self.bus),
            Arc::clone(&self.nodes),
            Arc::clone(&self.preds),
            Arc::clone(&state),
        );
        let finish = flow.emplace(move || {
            if last.failed.iter().any(|f| f.load(Ordering::Acquire)) {
                return;
            }
            let names: Vec<&'static str> = nodes.iter().map(|n| n.name).collect();
            if let Err(e) = bus.ledger().finish(&names, &preds) {
                last.fail(names.len(), e);
            }
        });
        flow.task(finish).name("budget: pack, accept or refuse");
        flow.task(finish).succeed(tasks.iter().copied());
        wait(executor, &flow)?;
        state.take_error(&self.rank).map_or(Ok(()), Err)
    }

    /// Execute mode: every step's `run` inside the wrapper, a step with a
    /// failed predecessor skipped and flagged failed in turn, so a failed
    /// step fails the steps that read it, directly or not, and nothing
    /// else. Allocates nothing budgeted. May be called again: a replay
    /// writes into the same buffers and the same scratch regions. In a
    /// binary that installs the [`AllocationRecord`], a step whose `run`
    /// made one allocation past [`RECORD_THRESHOLD`] on the thread running
    /// it fails, and the execute's peak heap growth is kept for the
    /// ledger's table ([`fb_core_budget::Ledger::measure_execute`]).
    ///
    /// # Errors
    /// `InvalidConfig` if the plan was not budgeted and accepted with bytes
    /// behind it; otherwise the first failing step's error in topological
    /// order, any other failure on its context; `InvalidConfig` for a step
    /// past the allocation record's threshold, naming it.
    pub fn execute(&self, executor: &Executor) -> Result<Vec<StepTime>> {
        let ledger = self.bus.ledger();
        if !ledger.sealed() || ledger.over() || ledger.backing() == Backing::Dry {
            return Err(Error::config(
                "execute: the plan was not budgeted and accepted with real allocations",
            ));
        }
        let state = RunState::new(self.nodes.iter().map(|n| n.name));
        let (flow, _) = self.flow("execute", |i| {
            let (bus, nodes, preds, state) = (
                Arc::clone(&self.bus),
                Arc::clone(&self.nodes),
                Arc::clone(&self.preds),
                Arc::clone(&state),
            );
            Box::new(move || {
                let name = nodes[i].name;
                if preds[i]
                    .iter()
                    .any(|&p| state.failed[p].load(Ordering::Acquire))
                {
                    state.failed[i].store(true, Ordering::Release);
                    log::debug!("step '{name}' skipped: a step it reads failed");
                    return;
                }
                let began = state.start.elapsed();
                match (nodes[i].run)(&bus) {
                    Ok((run, sync)) => {
                        let mut times = locked(&state.times);
                        times[i].began = began;
                        times[i].ended = state.start.elapsed();
                        times[i].run = run;
                        times[i].sync = sync;
                    }
                    Err(e) => state.fail(i, e.with_context(format!("step '{name}'"))),
                }
            })
        });
        // The plain heap's peak growth over the execute: the budget
        // table's unbudgeted row.
        ledger.measure_execute(|| wait(executor, &flow))?;
        if let Some(error) = state.take_error(&self.rank) {
            return Err(error);
        }
        Ok(locked(&state.times).clone())
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
