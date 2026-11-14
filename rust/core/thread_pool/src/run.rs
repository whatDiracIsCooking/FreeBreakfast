//! The executor: the one module that spawns threads.
#![allow(clippy::disallowed_methods)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use super::backend::Backend;
use super::plan::{Lane, NodeId, Placement, Plan};
use fb_core_error_handling::{Error, Status};
use fb_core_log::TaskScope;
use log::debug;
use serde::{Deserialize, Serialize};

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

/// One node's times in a [`Record`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskTime {
    /// The node.
    pub node: NodeId,
    /// Its name, as the plan gave it.
    pub name: String,
    /// The worker that ran it: `cpu<i>` or `stream<s>`, as its log lines
    /// and errors name it.
    pub worker: String,
    /// Its dimensions by name, as [`Plan::set_dims`] gave them; empty when
    /// none were.
    pub dims: BTreeMap<String, usize>,
    /// Wall time of its callable alone, not the waits before it: a CPU
    /// node's compute, a GPU node's enqueueing.
    #[serde(rename = "host_ns", with = "nanos")]
    pub host: Duration,
    /// A GPU node's stream scope, the device time between the events
    /// recorded on its stream before it ran and after it returned. `None`
    /// for a CPU node, under a backend with no device clock
    /// ([`HostOnly`](super::HostOnly)), and for a GPU node that failed or
    /// whose scope could not be read.
    #[serde(rename = "device_ns", with = "nanos_opt")]
    pub device: Option<Duration>,
}

/// A sync point of a run, and its host scope: a worker blocked on the
/// device until a GPU node's work completed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncPoint {
    /// The CPU node that waited before it ran, or `None` for a stream
    /// worker's sync as it exits.
    pub node: Option<NodeId>,
    /// The worker that waited: `cpu<i>` or `stream<s>`.
    pub worker: String,
    /// What it waited for: the GPU predecessor whose event the CPU node
    /// waited on, or the last node the exiting stream worker ran.
    pub waited_for: NodeId,
    /// Wall time of the wait.
    #[serde(rename = "host_ns", with = "nanos")]
    pub host: Duration,
}

/// One block of the plan a [`Record`] timed: a worker's arena of one kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaSize {
    /// The GPU worker it belongs to.
    pub worker: usize,
    /// Its kind's name: `device`, `pinned`, or `heap` in the tests.
    pub kind: String,
    /// Its size in bytes.
    pub bytes: usize,
}

/// What a [`run`] returns: its result and its timing record
/// (`docs/diagnostics.md`, "The timing record").
///
/// # `timing.json`
///
/// [`Record::to_json`] is the record as the `timing.json` text, one JSON
/// object; deserializing that text gives the record back. Times are integer
/// nanoseconds, in fields named `*_ns`; nodes are their ranks.
///
/// ```text
/// {
///   "error": null | {"kind": "Numerical", "message": "...", "context": ["name@worker", ...]},
///   "profile": "debug" | "release",
///   "wall_ns": 1234,
///   "tasks": [{"node": 0, "name": "S", "worker": "stream0", "dims": {"n": 24},
///              "host_ns": 10, "device_ns": 20 | null}, ...],
///   "syncs": [{"node": 3 | null, "worker": "cpu0", "waited_for": 2, "host_ns": 5}, ...],
///   "arena": [{"worker": 0, "kind": "device", "bytes": 4608}, ...]
/// }
/// ```
///
/// An error keeps its kind (the name of its
/// [`ErrorKind`](fb_core_error_handling::ErrorKind) variant), its message
/// and its context. Its origin, a source location in the binary that wrote
/// the file, is not written, and reads back as this module's.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The run's outcome: the failure lowest in rank, with `name@worker`
    /// and any other failures (`also: ...`) on its context.
    #[serde(rename = "error", with = "status")]
    pub result: Status,
    /// The profile that built the executor, `debug` or `release`
    /// (`cfg!(debug_assertions)`): a debug build's times are not compared
    /// with a release build's.
    pub profile: String,
    /// Wall time from the workers' start to their join, every stream
    /// synced: what the device times of one stream sum to at most.
    #[serde(rename = "wall_ns", with = "nanos")]
    pub wall: Duration,
    /// One entry per node that ran, in rank order, failed nodes included;
    /// a skipped node has none.
    pub tasks: Vec<TaskTime>,
    /// Every sync point of the run: each CPU node's wait on each GPU
    /// predecessor, by the node's rank and then the predecessor's, then
    /// each stream worker's sync as it exits, by the node it waited for. A
    /// stream worker that ran nothing waited for nothing, and has none.
    pub syncs: Vec<SyncPoint>,
    /// The plan's blocks, per worker and kind. Empty from [`run`], which
    /// sees no arena; the planner's `Instance::run` fills it.
    pub arena: Vec<ArenaSize>,
}

impl Record {
    fn new(result: Status) -> Self {
        Record {
            result,
            profile: profile().to_string(),
            wall: Duration::ZERO,
            tasks: Vec::new(),
            syncs: Vec::new(),
            arena: Vec::new(),
        }
    }

    /// The record as the `timing.json` text: the schema above, indented.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a record is plain data")
    }
}

/// The profile [`Record::profile`] names.
fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// A [`Duration`] as integer nanoseconds, which fit a `u64` for 584 years.
mod nanos {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub(super) fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_nanos() as u64)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        u64::deserialize(d).map(Duration::from_nanos)
    }
}

/// An `Option<Duration>` as integer nanoseconds or `null`.
mod nanos_opt {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub(super) fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match d {
            Some(d) => s.serialize_some(&(d.as_nanos() as u64)),
            None => s.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Duration>, D::Error> {
        Option::<u64>::deserialize(d).map(|n| n.map(Duration::from_nanos))
    }
}

/// A [`Status`] as `null` or the error's kind, message and context.
mod status {
    use std::panic::Location;

    use fb_core_error_handling::{Error, ErrorKind, Status};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Every kind, to read one back by name.
    const KINDS: [ErrorKind; 8] = [
        ErrorKind::Allocation,
        ErrorKind::Io,
        ErrorKind::GpuRuntime,
        ErrorKind::InvalidConfig,
        ErrorKind::Divergence,
        ErrorKind::MaxIterationsExceeded,
        ErrorKind::Unsupported,
        ErrorKind::Numerical,
    ];

    #[derive(Serialize, Deserialize)]
    struct Failed {
        kind: String,
        message: String,
        context: Vec<String>,
    }

    pub(super) fn serialize<S: Serializer>(status: &Status, s: S) -> Result<S::Ok, S::Error> {
        status
            .as_ref()
            .err()
            .map(|e| Failed {
                kind: format!("{:?}", e.kind),
                message: e.message.clone(),
                context: e.context.clone(),
            })
            .serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Status, D::Error> {
        let Some(failed) = Option::<Failed>::deserialize(d)? else {
            return Ok(Ok(()));
        };
        let kind = KINDS
            .into_iter()
            .find(|k| format!("{k:?}") == failed.kind)
            .ok_or_else(|| D::Error::custom(format!("unknown error kind {}", failed.kind)))?;
        Ok(Err(Error {
            kind,
            message: failed.message,
            origin: Location::caller(),
            context: failed.context,
        }))
    }
}

/// Run every node of `plan` on `n_cpu` CPU workers and `backend`'s stream
/// workers, and join them all before returning; see
/// [`thread_pool`](super) for what that promises. Every node is timed
/// whatever the outcome, so a failed run still returns the times of the
/// nodes that ran.
///
/// # Errors
/// In [`Record::result`]:
/// [`InvalidConfig`](fb_core_error_handling::ErrorKind::InvalidConfig)
/// before anything runs, if the plan places a node on a stream worker
/// `backend` lacks, or has CPU nodes and `n_cpu` is zero. Otherwise the
/// failure lowest in rank, with `name@worker` and any other failures
/// (`also: ...`) on its context.
///
/// # Panics
/// If a node panics, after every worker has stopped; see "Failure".
pub fn run<B: Backend>(plan: &Plan<'_, B::Stream>, backend: &B, n_cpu: usize) -> Record {
    let n_streams = backend.n_streams();
    if plan.n_streams() > n_streams {
        return Record::new(Err(Error::config(format!(
            "the plan places a node on stream worker {} of {n_streams}",
            plan.n_streams() - 1
        ))));
    }
    if n_cpu == 0 && plan.nodes.iter().any(|n| n.placement == Placement::Cpu) {
        return Record::new(Err(Error::config(
            "the plan has CPU nodes and no CPU workers",
        )));
    }
    if plan.is_empty() {
        return Record::new(Ok(()));
    }
    debug!(
        "run: {} nodes on {n_cpu} CPU and {n_streams} stream workers",
        plan.len()
    );
    let shared = Shared::new(plan, n_streams);
    let start = Instant::now();
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
    let wall = start.elapsed();
    let State {
        mut failures,
        mut tasks,
        mut syncs,
        scoped,
        ..
    } = shared
        .state
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    // Every stream is synced, so each closed stream scope reads without a
    // sync of its own. A read that fails after another failure is an echo
    // of it (a sticky fault reads back as itself), so only a run that
    // otherwise succeeded reports one, and only the first.
    tasks.sort_by_key(|t| t.node);
    for id in scoped {
        let at = tasks
            .binary_search_by_key(&NodeId(id), |t| t.node)
            .expect("a scoped node has an entry");
        match backend.elapsed(NodeId(id)) {
            Ok(device) => tasks[at].device = device,
            Err(error) if failures.is_empty() => {
                let task = &tasks[at];
                let context = format!("stream scope of {}@{}", task.name, task.worker);
                failures.push((id, error.with_context(context)));
            }
            Err(_) => {}
        }
    }
    debug!("run: done in {wall:?}, {} failed", failures.len());
    failures.sort_by_key(|(rank, _)| *rank);
    let mut failures = failures.into_iter().map(|(_, e)| e);
    let result = match failures.next() {
        None => Ok(()),
        Some(first) => {
            Err(failures.fold(first, |e, other| e.with_context(format!("also: {other}"))))
        }
    };
    syncs.sort_by_key(|s| (s.node.is_none(), s.node, s.waited_for));
    Record {
        wall,
        tasks,
        syncs,
        ..Record::new(result)
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
    /// Each node that ran, in the order they finished.
    tasks: Vec<TaskTime>,
    /// Each sync point, in the order they ended.
    syncs: Vec<SyncPoint>,
    /// The GPU nodes whose stream scope closed: those that succeeded.
    scoped: Vec<usize>,
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
            tasks: Vec::with_capacity(n),
            syncs: Vec::new(),
            scoped: Vec::new(),
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

    /// Node `id` finished with `result` after running for `time`, having
    /// waited at `syncs`: record them, and ready its successors or skip its
    /// descendants.
    fn finish<S>(&self, plan: &Plan<'_, S>, id: usize, (result, time, syncs): Ran) {
        let mut state = self.lock();
        state.syncs.extend(syncs);
        state.mark[id] = Mark::Done;
        state.remaining -= 1;
        if result.is_ok() && matches!(plan.nodes[id].placement, Placement::Stream(_)) {
            state.scoped.push(id);
        }
        state.tasks.push(time);
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
    let start = Instant::now();
    let synced = backend.sync(lane);
    let host = start.elapsed();
    if let Some(id) = last {
        shared.lock().syncs.push(SyncPoint {
            node: None,
            worker: name.to_string(),
            waited_for: NodeId(id),
            host,
        });
    }
    if let Err(error) = synced {
        let rank = last.unwrap_or(usize::MAX);
        let error = match last {
            Some(id) => error.with_context(format!("sync after {}@{name}", plan.nodes[id].name)),
            None => error.with_context(format!("sync@{name}")),
        };
        shared.lock().failures.push((rank, error));
    }
}

/// What [`execute`] hands [`Shared::finish`]: the node's result, its entry,
/// and the sync points it waited at.
type Ran = (Status, TaskTime, Vec<SyncPoint>);

/// Wait for the node's GPU predecessors as its edges need, and run it inside
/// its scopes: wall time around each host wait and around the callable, and
/// for a GPU node the stream scope, whose closing event is the one its
/// consumers wait on.
fn execute<B: Backend>(
    plan: &Plan<'_, B::Stream>,
    backend: &B,
    id: usize,
    worker: &str,
    lane: Option<usize>,
) -> Ran {
    let node = &plan.nodes[id];
    let _task = TaskScope::enter(&node.name, worker);
    debug!("start");
    let mut host = Duration::ZERO;
    let mut syncs = Vec::new();
    let result = (|| {
        for &p in &node.preds {
            let Placement::Stream(from) = plan.nodes[p].placement else {
                continue;
            };
            match lane {
                None => {
                    let start = Instant::now();
                    let waited = backend.host_wait(NodeId(p));
                    syncs.push(SyncPoint {
                        node: Some(NodeId(id)),
                        worker: worker.to_string(),
                        waited_for: NodeId(p),
                        host: start.elapsed(),
                    });
                    waited?;
                }
                Some(to) if to != from => backend.wait(to, NodeId(p))?,
                Some(_) => {}
            }
        }
        match lane {
            None => {
                let start = Instant::now();
                let result = (node.run)(Lane::Cpu);
                host = start.elapsed();
                result
            }
            Some(index) => {
                backend.begin(index, NodeId(id))?;
                let start = Instant::now();
                let result = (node.run)(Lane::Stream {
                    index,
                    stream: backend.stream(index),
                });
                host = start.elapsed();
                result?;
                backend.record(index, NodeId(id))
            }
        }
    })();
    debug!(
        "end in {host:?}{}",
        if result.is_ok() { "" } else { ": failed" }
    );
    let time = TaskTime {
        node: NodeId(id),
        name: node.name.clone(),
        worker: worker.to_string(),
        dims: node.dims.clone(),
        host,
        device: None,
    };
    (
        result.map_err(|e| e.with_context(format!("{}@{worker}", node.name))),
        time,
        syncs,
    )
}
