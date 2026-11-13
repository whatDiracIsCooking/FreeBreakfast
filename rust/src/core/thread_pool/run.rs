//! The executor: the one module that spawns threads.
#![allow(clippy::disallowed_methods)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

use super::backend::Backend;
use super::plan::{Lane, NodeId, Placement, Plan};
use crate::core::diagnostics::debug;
use crate::core::error_handling::{Error, Status};
use crate::core::log::TaskScope;

/// The default CPU worker count beside `n_streams` stream workers: the
/// host's available parallelism less the stream workers' threads, and at
/// least one.
pub fn default_n_cpu(n_streams: usize) -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .saturating_sub(n_streams)
        .max(1)
}

/// Run every node of `plan` on `n_cpu` CPU workers and `backend`'s stream
/// workers, and join them all before returning; see
/// [`thread_pool`](super) for what that promises.
///
/// # Errors
/// [`InvalidConfig`](crate::core::error_handling::ErrorKind::InvalidConfig)
/// before anything runs, if the plan places a node on a stream worker
/// `backend` lacks, or has CPU nodes and `n_cpu` is zero. Otherwise the
/// failure lowest in rank, with `name@worker` and any other failures
/// (`also: ...`) on its context.
///
/// # Panics
/// If a node panics, after every worker has stopped; see "Failure".
pub fn run<B: Backend>(plan: &Plan<'_, B::Stream>, backend: &B, n_cpu: usize) -> Status {
    let n_streams = backend.n_streams();
    if plan.n_streams() > n_streams {
        return Err(Error::config(format!(
            "the plan places a node on stream worker {} of {n_streams}",
            plan.n_streams() - 1
        )));
    }
    if n_cpu == 0 && plan.nodes.iter().any(|n| n.placement == Placement::Cpu) {
        return Err(Error::config("the plan has CPU nodes and no CPU workers"));
    }
    if plan.is_empty() {
        return Ok(());
    }
    debug!(
        "run: {} nodes on {n_cpu} CPU and {n_streams} stream workers",
        plan.len()
    );
    let shared = Shared::new(plan, n_streams);
    thread::scope(|scope| {
        let shared = &shared;
        let mut workers = Vec::with_capacity(n_cpu + n_streams);
        for i in 0..n_cpu {
            workers.push(spawn(scope, format!("cpu{i}"), move |name| {
                cpu_worker(plan, backend, shared, name)
            }));
        }
        for lane in 0..n_streams {
            workers.push(spawn(scope, format!("stream{lane}"), move |name| {
                stream_worker(plan, backend, shared, name, lane)
            }));
        }
        // Joined here rather than by the scope, so a node's panic is
        // re-raised with its own message, not the scope's generic one.
        let panics: Vec<_> = workers.into_iter().filter_map(|w| w.join().err()).collect();
        if let Some(panic) = panics.into_iter().next() {
            std::panic::resume_unwind(panic);
        }
    });
    let mut failures = shared
        .state
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .failures;
    debug!("run: done, {} failed", failures.len());
    failures.sort_by_key(|(rank, _)| *rank);
    let mut failures = failures.into_iter().map(|(_, e)| e);
    match failures.next() {
        None => Ok(()),
        Some(first) => {
            Err(failures.fold(first, |e, other| e.with_context(format!("also: {other}"))))
        }
    }
}

fn spawn<'scope, 'env>(
    scope: &'scope thread::Scope<'scope, 'env>,
    name: String,
    body: impl FnOnce(&str) + Send + 'scope,
) -> thread::ScopedJoinHandle<'scope, ()> {
    thread::Builder::new()
        .name(format!("fb-{name}"))
        .spawn_scoped(scope, move || body(&name))
        .expect("the OS refused a worker thread the plan admitted")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    Pending,
    Done,
    Skipped,
}

#[derive(Clone, Copy)]
enum Queue {
    Cpu,
    Lane(usize),
}

struct State {
    /// Per node, its predecessors not yet finished.
    waiting: Vec<usize>,
    mark: Vec<Mark>,
    cpu_ready: BinaryHeap<Reverse<usize>>,
    lane_ready: Vec<BinaryHeap<Reverse<usize>>>,
    /// Nodes neither finished nor skipped.
    remaining: usize,
    /// Each failure with the rank it is reported at.
    failures: Vec<(usize, Error)>,
    /// A worker panicked: everyone stops.
    aborted: bool,
}

struct Shared {
    state: Mutex<State>,
    cpu: Condvar,
    lanes: Vec<Condvar>,
}

impl Shared {
    fn new<S>(plan: &Plan<'_, S>, n_streams: usize) -> Self {
        let n = plan.len();
        let mut state = State {
            waiting: plan.nodes.iter().map(|node| node.preds.len()).collect(),
            mark: vec![Mark::Pending; n],
            cpu_ready: BinaryHeap::new(),
            lane_ready: (0..n_streams).map(|_| BinaryHeap::new()).collect(),
            remaining: n,
            failures: Vec::new(),
            aborted: false,
        };
        for (id, node) in plan.nodes.iter().enumerate() {
            if node.preds.is_empty() {
                state.queue(node.placement).push(Reverse(id));
            }
        }
        Shared {
            state: Mutex::new(state),
            cpu: Condvar::new(),
            lanes: (0..n_streams).map(|_| Condvar::new()).collect(),
        }
    }

    /// The lock, through a poisoning: a worker that panicked holding it set
    /// nothing half-way that the others read, and they must still see
    /// `aborted` to stop.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn condvar(&self, queue: Queue) -> &Condvar {
        match queue {
            Queue::Cpu => &self.cpu,
            Queue::Lane(l) => &self.lanes[l],
        }
    }

    fn wake_all(&self) {
        self.cpu.notify_all();
        self.lanes.iter().for_each(Condvar::notify_all);
    }

    /// The next node for `queue`, waiting for one; `None` when the run is
    /// over.
    fn next(&self, queue: Queue) -> Option<usize> {
        let mut state = self.lock();
        loop {
            if state.aborted || state.remaining == 0 {
                return None;
            }
            let ready = match queue {
                Queue::Cpu => &mut state.cpu_ready,
                Queue::Lane(l) => &mut state.lane_ready[l],
            };
            if let Some(Reverse(id)) = ready.pop() {
                return Some(id);
            }
            state = self
                .condvar(queue)
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Node `id` finished with `result`: ready its successors, or skip its
    /// descendants.
    fn finish<S>(&self, plan: &Plan<'_, S>, id: usize, result: Status) {
        let mut state = self.lock();
        state.mark[id] = Mark::Done;
        state.remaining -= 1;
        match result {
            Ok(()) => {
                for &s in &plan.succs[id] {
                    if state.mark[s] != Mark::Pending {
                        continue;
                    }
                    state.waiting[s] -= 1;
                    if state.waiting[s] == 0 {
                        let placement = plan.nodes[s].placement;
                        state.queue(placement).push(Reverse(s));
                        self.condvar(Queue::from(placement)).notify_one();
                    }
                }
            }
            Err(error) => {
                state.failures.push((id, error));
                let mut stack = plan.succs[id].clone();
                while let Some(s) = stack.pop() {
                    if state.mark[s] == Mark::Pending {
                        state.mark[s] = Mark::Skipped;
                        state.remaining -= 1;
                        stack.extend_from_slice(&plan.succs[s]);
                    }
                }
            }
        }
        if state.remaining == 0 {
            drop(state);
            self.wake_all();
        }
    }
}

impl State {
    fn queue(&mut self, placement: Placement) -> &mut BinaryHeap<Reverse<usize>> {
        match placement {
            Placement::Cpu => &mut self.cpu_ready,
            Placement::Stream(l) => &mut self.lane_ready[l],
        }
    }
}

impl From<Placement> for Queue {
    fn from(placement: Placement) -> Self {
        match placement {
            Placement::Cpu => Queue::Cpu,
            Placement::Stream(l) => Queue::Lane(l),
        }
    }
}

/// While its worker unwinds from a panic: stop the run and wake everyone,
/// so the scope can join and re-raise the panic instead of hanging.
struct AbortOnUnwind<'s>(&'s Shared);

impl Drop for AbortOnUnwind<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.lock().aborted = true;
            self.0.wake_all();
        }
    }
}

fn cpu_worker<B: Backend>(plan: &Plan<'_, B::Stream>, backend: &B, shared: &Shared, name: &str) {
    let _abort = AbortOnUnwind(shared);
    while let Some(id) = shared.next(Queue::Cpu) {
        let result = execute(plan, backend, id, name, None);
        shared.finish(plan, id, result);
    }
}

fn stream_worker<B: Backend>(
    plan: &Plan<'_, B::Stream>,
    backend: &B,
    shared: &Shared,
    name: &str,
    lane: usize,
) {
    let _abort = AbortOnUnwind(shared);
    let _binding = backend.bind(lane);
    let mut last = None;
    while let Some(id) = shared.next(Queue::Lane(lane)) {
        let result = execute(plan, backend, id, name, Some(lane));
        shared.finish(plan, id, result);
        last = Some(id);
    }
    // The end of the run is a sync point: nothing is left enqueued, and a
    // fault surfacing here is the stream's, reported against its last node.
    if let Err(error) = backend.sync(lane) {
        let rank = last.unwrap_or(usize::MAX);
        let error = match last {
            Some(id) => error.with_context(format!("sync after {}@{name}", plan.nodes[id].name)),
            None => error.with_context(format!("sync@{name}")),
        };
        shared.lock().failures.push((rank, error));
    }
}

/// Wait for the node's GPU predecessors as its edges need, run it, and
/// record its event if it is a GPU node.
fn execute<B: Backend>(
    plan: &Plan<'_, B::Stream>,
    backend: &B,
    id: usize,
    worker: &str,
    lane: Option<usize>,
) -> Status {
    let node = &plan.nodes[id];
    let _task = TaskScope::enter(&node.name, worker);
    debug!("start");
    let result = (|| {
        for &p in &node.preds {
            let Placement::Stream(from) = plan.nodes[p].placement else {
                continue;
            };
            match lane {
                None => backend.host_wait(NodeId(p))?,
                Some(to) if to != from => backend.wait(to, NodeId(p))?,
                Some(_) => {}
            }
        }
        match lane {
            None => (node.run)(Lane::Cpu),
            Some(index) => {
                (node.run)(Lane::Stream {
                    index,
                    stream: backend.stream(index),
                })?;
                backend.record(index, NodeId(id))
            }
        }
    })();
    debug!("end{}", if result.is_ok() { "" } else { ": failed" });
    result.map_err(|e| e.with_context(format!("{}@{worker}", node.name)))
}
