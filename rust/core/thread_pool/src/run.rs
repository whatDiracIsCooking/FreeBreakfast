//! The executor: the one module that spawns threads.
#![allow(clippy::disallowed_methods)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::ops::ControlFlow;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use super::backend::Backend;
use super::plan::{Lane, LaneId, LoopKind, NodeId, Placement, Plan, Work};
use fb_core_error_handling::{Error, Status};
use fb_core_log::TaskScope;
use log::debug;
use serde::{Deserialize, Serialize};

/// The default CPU worker count beside `n_gpu` GPU workers: the host's
/// available parallelism less the GPU workers' threads, one each whatever
/// its lanes, and at least one.
pub fn default_n_cpu(n_gpu: usize) -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .saturating_sub(n_gpu)
        .max(1)
}

/// One node's times in a [`Record`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskTime {
    /// The node.
    pub node: NodeId,
    /// Its name, as the plan gave it.
    pub name: String,
    /// The worker that ran it: `cpu<i>`, or for a GPU node its lane,
    /// `gpu<w>.<l>` ([`LaneId`](super::LaneId)), as its log lines and
    /// errors name it.
    pub worker: String,
    /// Its dimensions by name, as [`Plan::set_dims`] gave them; empty when
    /// none were.
    pub dims: BTreeMap<String, usize>,
    /// For a loop's body node or decision, the iteration it ran in, from 0:
    /// such a node has one entry per iteration, a counted loop's body node
    /// one per trip. `None` for a node outside every loop.
    pub iteration: Option<usize>,
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
    /// The CPU node that waited before it ran, or `None` for a GPU
    /// worker's sync of one of its lanes as it exits.
    pub node: Option<NodeId>,
    /// The iteration that node ran in, when it is a loop's body node or
    /// decision; `None` otherwise.
    pub iteration: Option<usize>,
    /// What waited: the CPU worker `cpu<i>`, or the lane `gpu<w>.<l>` its
    /// GPU worker synced.
    pub worker: String,
    /// What it waited for: the GPU predecessor whose event the CPU node
    /// waited on, or the last node the exiting GPU worker ran on the lane.
    pub waited_for: NodeId,
    /// Wall time of the wait.
    #[serde(rename = "host_ns", with = "nanos")]
    pub host: Duration,
}

/// One block of the plan a [`Record`] timed: a worker's arena of one kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaSize {
    /// The GPU worker it belongs to; 0 for the plan's heap block, which
    /// is every worker's.
    pub worker: usize,
    /// Its kind's name: `device`, `pinned` or `heap`.
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
///   "tasks": [{"node": 0, "name": "S", "worker": "gpu0.0", "dims": {"n": 24},
///              "iteration": null | 0, "host_ns": 10, "device_ns": 20 | null}, ...],
///   "syncs": [{"node": 3 | null, "iteration": null | 0, "worker": "cpu0",
///              "waited_for": 2, "host_ns": 5}, ...],
///   "iterations": {"7": 12, ...},
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
    /// Wall time from the workers' start to their join, every lane
    /// synced: what the device times of one lane sum to at most.
    #[serde(rename = "wall_ns", with = "nanos")]
    pub wall: Duration,
    /// One entry per node that ran, by rank and then iteration, failed
    /// nodes included: a loop's body nodes and decision have one per
    /// iteration they ran in. A skipped node has none.
    pub tasks: Vec<TaskTime>,
    /// Every sync point of the run: each CPU node's wait on each GPU
    /// predecessor, by the node's rank, its iteration and then the
    /// predecessor's rank, then each GPU worker's sync of each lane as it
    /// exits, by the node it waited for. A lane that ran nothing is not
    /// synced, and has none. An edge between two lanes is a wait on the
    /// device, and has none either.
    pub syncs: Vec<SyncPoint>,
    /// Each decision loop that began, by its decision node
    /// ([`Plan::add_loop`]): the number of iterations its body began. A
    /// counted loop's count is its plan's, and is not here. For a loop that stopped, its
    /// decision's last index plus one; for one that failed, the iteration
    /// it failed in plus one. In `timing.json`, the keys are the ranks as
    /// strings.
    pub iterations: BTreeMap<NodeId, usize>,
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
            iterations: BTreeMap::new(),
            arena: Vec::new(),
        }
    }

    /// The record as the `timing.json` text: the schema above, indented.
    ///
    /// # Panics
    /// Never in practice: a record is plain data, which `serde_json` always
    /// writes.
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

/// Run every node of `plan` on `n_cpu` CPU workers and `backend`'s GPU
/// workers, and join them all before returning; see
/// [`thread_pool`](super) for what that promises. Every node is timed
/// whatever the outcome, so a failed run still returns the times of the
/// nodes that ran.
///
/// # Errors
/// In [`Record::result`]:
/// [`InvalidConfig`](fb_core_error_handling::ErrorKind::InvalidConfig)
/// before anything runs, if the plan places a node on a lane `backend`
/// lacks, or has CPU nodes and `n_cpu` is zero. Otherwise the
/// failure lowest in rank, with `name@worker` and any other failures
/// (`also: ...`) on its context: a node's own error, or
/// [`MaxIterationsExceeded`](fb_core_error_handling::ErrorKind::MaxIterationsExceeded)
/// from a loop's decision that would pass its bound.
///
/// Both are borrowed exclusively, so a plan and a backend are each in at
/// most one run at a time: a backend's events are its plan's and its
/// streams and handles are its lanes', and a plan's nodes reach
/// whatever its closures captured, an instance's blocks among them
/// ([`thread_pool`](super), "One run at a time"). Inside the run the
/// workers share both by reference.
///
/// # Panics
/// If a node panics, after every worker has stopped; see "Failure".
pub fn run<B: Backend>(plan: &mut Plan<'_, B::Stream>, backend: &mut B, n_cpu: usize) -> Record {
    let (plan, backend) = (&*plan, &*backend);
    let n_gpu = backend.n_workers();
    if let Some(lane) = plan.nodes.iter().find_map(|n| match n.placement {
        Placement::Stream(at) if !backend.has_lane(at) => Some(at),
        _ => None,
    }) {
        return Record::new(Err(Error::config(format!(
            "the plan places a node on lane {lane}, which the backend's \
             {n_gpu} GPU workers lack"
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
        "run: {} nodes on {n_cpu} CPU and {n_gpu} GPU workers",
        plan.len()
    );
    let graph = Graph::new(plan);
    let shared = Shared::new(plan, &graph, n_gpu);
    let backend = &RunBackend(backend);
    let start = Instant::now();
    thread::scope(|scope| {
        let shared = &shared;
        let mut workers = Vec::with_capacity(n_cpu + n_gpu);
        let graph = &graph;
        for i in 0..n_cpu {
            workers.push(spawn(scope, format!("cpu{i}"), move |name| {
                cpu_worker((plan, graph), &**backend, shared, name)
            }));
        }
        for worker in 0..n_gpu {
            workers.push(spawn(scope, format!("gpu{worker}"), move |_| {
                gpu_worker((plan, graph), &**backend, shared, worker)
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
        began,
        ..
    } = shared
        .state
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    // Every stream is synced, so each closed stream scope reads without a
    // sync of its own. A read that fails after another failure is an echo
    // of it (a sticky fault reads back as itself), so only a run that
    // otherwise succeeded reports one, and only the first.
    tasks.sort_by_key(|t| (t.node, t.iteration));
    for (id, iteration, trip) in scoped {
        let at = tasks
            .binary_search_by_key(&(NodeId(id), iteration), |t| (t.node, t.iteration))
            .expect("a scoped node has an entry");
        match backend.elapsed(NodeId(id), trip) {
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
    syncs.sort_by_key(|s| (s.node.is_none(), s.node, s.iteration, s.waited_for));
    let iterations = plan
        .loops
        .iter()
        .zip(began)
        .filter(|&(_, n)| n > 0)
        .filter_map(|(l, n)| Some((NodeId(l.decision()?), n)))
        .collect();
    Record {
        wall,
        tasks,
        syncs,
        iterations,
        ..Record::new(result)
    }
}

/// The backend as one run's workers share it: [`run`] makes the only one,
/// from its exclusive borrow, and it lives no longer than the run's scope.
/// A backend need not be `Sync` (a `Device` is not); this
/// is where sharing it across threads is argued, once, for one run.
struct RunBackend<'a, B>(&'a B);

// SAFETY: `Backend` is sealed, so `B` is `HostOnly` (plain data, `Sync`
// itself) or `Device`. A `RunBackend` exists only inside `run`, made from
// its `&mut` borrow, so no other run, and no code outside this one, holds
// the backend meanwhile; its threads are this run's workers, which call
// each method only from the thread its doc names. For a `Device`, whose
// stream, event and library handles are raw runtime pointers: the runtime
// API is thread-safe for every call made here through a shared reference
// (record, wait, synchronize, elapsed time), on events that are this plan's
// and no other run's. A library handle is called through only by the nodes
// of its own lane, one node at a time on its GPU worker's thread, and cuBLAS,
// hipBLAS, cuSOLVER and hipSOLVER allow a handle to be used from any thread
// one call at a time; a `DeviceStream` is not `Sync`, so a node cannot hand
// its stream or handles to another thread. Every handle and every `Rc` a
// `DeviceStream` holds is made before the run and dropped after it, and a
// BLAS workspace is handed over only through `&mut Device`, which the run's
// borrow rules out.
unsafe impl<B: Backend> Sync for RunBackend<'_, B> {}

impl<B> std::ops::Deref for RunBackend<'_, B> {
    type Target = B;
    fn deref(&self) -> &B {
        self.0
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

/// The run's unit of scheduling, an instance: a node in one trip of the
/// counted loop it is in, or once for a node outside every counted loop.
/// Instances are numbered in rank order with each counted loop's body
/// unrolled iteration by iteration, which is a topological order of
/// their edges: each node's own, taken within one trip of a counted body
/// and from a closed counted body's last trip after it; each counted body
/// node's after itself in the trip before; and the ring, each source of
/// trip `k` of a counted body after each of its sinks in trip
/// `k - depth`. A decision loop's body is a run of instances its decision
/// readies again. Built once per run, before the workers start.
struct Graph {
    /// Per instance, its node.
    node: Vec<usize>,
    /// Per instance, its trip: 0 outside a counted loop.
    trip: Vec<usize>,
    preds: Vec<Vec<usize>>,
    succs: Vec<Vec<usize>>,
    /// Per node, its instance in trip 0.
    first: Vec<usize>,
}

impl Graph {
    fn new<S>(plan: &Plan<'_, S>) -> Self {
        let n = plan.len();
        let counted = |p: usize| match plan.nodes[p].in_loop.map(|l| &plan.loops[l]) {
            Some(l) => match l.kind {
                LoopKind::Counted { trips, depth } => Some((l.body.clone(), trips, depth)),
                LoopKind::Decide { .. } => None,
            },
            None => None,
        };
        let (mut node, mut trip, mut first) = (Vec::new(), Vec::new(), vec![0; n]);
        let mut r = 0;
        while r < n {
            let Some((body, trips, _)) = counted(r) else {
                first[r] = node.len();
                node.push(r);
                trip.push(0);
                r += 1;
                continue;
            };
            for k in 0..trips {
                for p in body.clone() {
                    if k == 0 {
                        first[p] = node.len();
                    }
                    node.push(p);
                    trip.push(k);
                }
            }
            r = body.end;
        }
        // Node `p`'s instance in trip `k`, and its last trip.
        let at = |p: usize, k: usize| first[p] + counted(p).map_or(0, |(b, _, _)| k * b.len());
        let last = |p: usize| counted(p).map_or(0, |(_, trips, _)| trips - 1);
        let mut preds: Vec<Vec<usize>> = (0..node.len())
            .map(|i| {
                let (p, k) = (node[i], trip[i]);
                let body = counted(p);
                let inside = |q: &usize| body.as_ref().is_some_and(|(b, _, _)| b.contains(q));
                let mut preds: Vec<usize> = plan.nodes[p]
                    .preds
                    .iter()
                    .map(|&q| if inside(&q) { at(q, k) } else { at(q, last(q)) })
                    .collect();
                if let Some((b, _, depth)) = &body {
                    if k >= 1 {
                        preds.push(at(p, k - 1));
                    }
                    if k >= *depth && !plan.nodes[p].preds.iter().any(inside) {
                        let sinks = b.clone().filter(|&s| !plan.succs[s].iter().any(inside));
                        preds.extend(sinks.map(|s| at(s, k - depth)));
                    }
                }
                preds
            })
            .collect();
        let mut succs = vec![Vec::new(); node.len()];
        for (i, ps) in preds.iter_mut().enumerate() {
            ps.sort_unstable();
            ps.dedup();
            for &q in ps.iter() {
                succs[q].push(i);
            }
        }
        Graph {
            node,
            trip,
            preds,
            succs,
            first,
        }
    }

    fn len(&self) -> usize {
        self.node.len()
    }
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
    /// A GPU worker's, every lane of it in one rank order.
    Gpu(usize),
}

struct State {
    /// Per instance, its predecessors not yet finished.
    waiting: Vec<usize>,
    mark: Vec<Mark>,
    cpu_ready: BinaryHeap<Reverse<usize>>,
    gpu_ready: Vec<BinaryHeap<Reverse<usize>>>,
    /// Instances neither finished nor skipped.
    remaining: usize,
    /// Each failure with the rank it is reported at.
    failures: Vec<(usize, Error)>,
    /// Each node that ran, in the order they finished.
    tasks: Vec<TaskTime>,
    /// Each sync point, in the order they ended.
    syncs: Vec<SyncPoint>,
    /// The GPU nodes whose stream scope closed, those that succeeded, each
    /// with its iteration and trip, and not yet read: a decision loop's
    /// are read by its decision before the body is replayed.
    scoped: Vec<(usize, Option<usize>, usize)>,
    /// Per loop, the index of the iteration a decision loop's body is in.
    iteration: Vec<usize>,
    /// Per loop, the iterations a decision loop's body began.
    began: Vec<usize>,
    /// A worker panicked: everyone stops.
    aborted: bool,
}

struct Shared {
    state: Mutex<State>,
    cpu: Condvar,
    gpus: Vec<Condvar>,
}

/// An instance a worker took to run: the instance, its node, the
/// iteration it runs in (`None` outside every loop) and its trip.
#[derive(Clone, Copy)]
struct Job {
    instance: usize,
    node: usize,
    iteration: Option<usize>,
    trip: usize,
}

impl Shared {
    fn new<S>(plan: &Plan<'_, S>, graph: &Graph, n_gpu: usize) -> Self {
        let n = graph.len();
        let mut state = State {
            waiting: graph.preds.iter().map(Vec::len).collect(),
            mark: vec![Mark::Pending; n],
            cpu_ready: BinaryHeap::new(),
            gpu_ready: (0..n_gpu).map(|_| BinaryHeap::new()).collect(),
            remaining: n,
            failures: Vec::new(),
            tasks: Vec::with_capacity(n),
            syncs: Vec::new(),
            scoped: Vec::new(),
            iteration: vec![0; plan.loops.len()],
            began: vec![0; plan.loops.len()],
            aborted: false,
        };
        for (i, preds) in graph.preds.iter().enumerate() {
            if preds.is_empty() {
                let placement = plan.nodes[graph.node[i]].placement;
                state.queue(placement).push(Reverse(i));
            }
        }
        Shared {
            state: Mutex::new(state),
            cpu: Condvar::new(),
            gpus: (0..n_gpu).map(|_| Condvar::new()).collect(),
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
            Queue::Gpu(w) => &self.gpus[w],
        }
    }

    fn wake_all(&self) {
        self.cpu.notify_all();
        self.gpus.iter().for_each(Condvar::notify_all);
    }

    /// The next instance for `queue`, waiting for one; `None` when the run
    /// is over.
    fn next<S>(&self, (plan, graph): (&Plan<'_, S>, &Graph), queue: Queue) -> Option<Job> {
        let mut state = self.lock();
        loop {
            if state.aborted || state.remaining == 0 {
                return None;
            }
            let ready = match queue {
                Queue::Cpu => &mut state.cpu_ready,
                Queue::Gpu(w) => &mut state.gpu_ready[w],
            };
            if let Some(Reverse(instance)) = ready.pop() {
                let (node, trip) = (graph.node[instance], graph.trip[instance]);
                let iteration = plan.nodes[node].in_loop.map(|l| {
                    match plan.loops[l].kind {
                        // Fixed until the node finishes: its decision is after it.
                        LoopKind::Decide { .. } => {
                            state.began[l] = state.iteration[l] + 1;
                            state.iteration[l]
                        }
                        LoopKind::Counted { .. } => trip,
                    }
                });
                return Some(Job {
                    instance,
                    node,
                    iteration,
                    trip,
                });
            }
            state = self
                .condvar(queue)
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// `job` finished as [`Ran`] says: record its entry and sync points,
    /// and replay its loop's body, ready its successors or skip its
    /// descendants.
    fn finish<S>(&self, (plan, graph): (&Plan<'_, S>, &Graph), job: Job, ran: Ran) {
        let Ran {
            result,
            time,
            syncs,
            replay,
            scopes,
        } = ran;
        let Job {
            instance,
            node: id,
            iteration,
            trip,
        } = job;
        let mut state = self.lock();
        state.syncs.extend(syncs);
        state.mark[instance] = Mark::Done;
        state.remaining -= 1;
        if result.is_ok() && matches!(plan.nodes[id].placement, Placement::Stream(_)) {
            state.scoped.push((id, iteration, trip));
        }
        state.tasks.push(time);
        if let Work::Decide(l, _) = plan.nodes[id].work
            && result.is_ok()
        {
            // The decision read every body scope of this iteration.
            let body = &plan.loops[l].body;
            state.scoped.retain(|(n, _, _)| !body.contains(n));
            let iteration = Some(state.iteration[l]);
            for (node, device) in scopes {
                let task = state
                    .tasks
                    .iter_mut()
                    .rev()
                    .find(|t| t.node.0 == node && t.iteration == iteration)
                    .expect("a scoped node has an entry");
                task.device = Some(device);
            }
        }
        match result {
            Ok(()) if replay => {
                let Work::Decide(l, _) = plan.nodes[id].work else {
                    unreachable!("only a decision replays")
                };
                let lp = &plan.loops[l];
                state.iteration[l] += 1;
                debug!("{}: iteration {}", plan.nodes[id].name, state.iteration[l]);
                // A decision loop's body holds no counted loop, so its
                // instances are a run ending at the decision's.
                let start = graph.first[lp.body.start];
                state.remaining += instance + 1 - start;
                for i in start..=instance {
                    // Its predecessors before the body ran once, for good.
                    let waiting = graph.preds[i].iter().filter(|&&q| q >= start).count();
                    state.mark[i] = Mark::Pending;
                    state.waiting[i] = waiting;
                    if waiting == 0 {
                        let placement = plan.nodes[graph.node[i]].placement;
                        state.queue(placement).push(Reverse(i));
                        self.condvar(Queue::from(placement)).notify_one();
                    }
                }
            }
            Ok(()) => {
                for &s in &graph.succs[instance] {
                    if state.mark[s] != Mark::Pending {
                        continue;
                    }
                    state.waiting[s] -= 1;
                    if state.waiting[s] == 0 {
                        let placement = plan.nodes[graph.node[s]].placement;
                        state.queue(placement).push(Reverse(s));
                        self.condvar(Queue::from(placement)).notify_one();
                    }
                }
            }
            Err(error) => {
                state.failures.push((id, error));
                let mut stack = graph.succs[instance].clone();
                while let Some(s) = stack.pop() {
                    if state.mark[s] == Mark::Pending {
                        state.mark[s] = Mark::Skipped;
                        state.remaining -= 1;
                        stack.extend_from_slice(&graph.succs[s]);
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
            Placement::Stream(at) => &mut self.gpu_ready[at.worker],
        }
    }
}

impl From<Placement> for Queue {
    fn from(placement: Placement) -> Self {
        match placement {
            Placement::Cpu => Queue::Cpu,
            Placement::Stream(at) => Queue::Gpu(at.worker),
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

/// What a worker reads the run through: the plan and its instances.
type Run<'r, 'a, S> = (&'r Plan<'a, S>, &'r Graph);

fn cpu_worker<B: Backend>(run: Run<'_, '_, B::Stream>, backend: &B, shared: &Shared, name: &str) {
    let _abort = AbortOnUnwind(shared);
    while let Some(job) = shared.next(run, Queue::Cpu) {
        let ran = execute(run, backend, job, name, None);
        shared.finish(run, job, ran);
    }
}

/// GPU worker `worker`: its device current for the whole run, it enqueues
/// its ready nodes in rank order across all its lanes, each on its lane's
/// stream, a counted loop's iterations back to back, and never waits on
/// the device until it exits.
fn gpu_worker<B: Backend>(
    run: Run<'_, '_, B::Stream>,
    backend: &B,
    shared: &Shared,
    worker: usize,
) {
    let _abort = AbortOnUnwind(shared);
    let _binding = backend.bind(worker);
    let plan = run.0;
    // Per lane that ran a node, the last one.
    let mut last: BTreeMap<LaneId, usize> = BTreeMap::new();
    while let Some(job) = shared.next(run, Queue::Gpu(worker)) {
        let Placement::Stream(lane) = plan.nodes[job.node].placement else {
            unreachable!("a GPU worker's queue holds its lanes' nodes")
        };
        let ran = execute(run, backend, job, &lane.to_string(), Some(lane));
        shared.finish(run, job, ran);
        last.insert(lane, job.node);
    }
    // The end of the run is a sync point on each lane: nothing is left
    // enqueued, and a fault surfacing here is the lane's, reported against
    // its last node.
    for (lane, id) in last {
        let name = lane.to_string();
        let start = Instant::now();
        let synced = backend.sync(lane);
        let host = start.elapsed();
        let mut state = shared.lock();
        state.syncs.push(SyncPoint {
            node: None,
            iteration: None,
            worker: name.clone(),
            waited_for: NodeId(id),
            host,
        });
        if let Err(error) = synced {
            let error = error.with_context(format!("sync after {}@{name}", plan.nodes[id].name));
            state.failures.push((id, error));
        }
    }
}

/// What [`execute`] hands [`Shared::finish`].
struct Ran {
    /// The node's result.
    result: Status,
    /// Its entry.
    time: TaskTime,
    /// The sync points it waited at.
    syncs: Vec<SyncPoint>,
    /// A decision that said `Continue` within its bound.
    replay: bool,
    /// From a decision: the device time of each GPU body node of its
    /// iteration, read once the whole body had completed.
    scopes: Vec<(usize, Duration)>,
}

/// Wait for the instance's GPU predecessors as its edges need, and run it
/// inside its scopes: wall time around each host wait and around the
/// callable, and for a GPU node the stream scope of its trip, whose closing
/// event is the one its consumers wait on. A decision, once its waits have
/// made its whole body complete, first reads the body's stream scopes of
/// this iteration, before the replay records them again.
fn execute<B: Backend>(
    (plan, graph): Run<'_, '_, B::Stream>,
    backend: &B,
    job: Job,
    worker: &str,
    lane: Option<LaneId>,
) -> Ran {
    let Job {
        instance,
        node: id,
        iteration,
        trip,
    } = job;
    let node = &plan.nodes[id];
    let _task = TaskScope::enter(&node.name, worker);
    debug!("start");
    let mut host = Duration::ZERO;
    let mut syncs = Vec::new();
    let mut replay = false;
    let mut scopes = Vec::new();
    let result = (|| {
        for &q in &graph.preds[instance] {
            let (p, from_trip) = (graph.node[q], graph.trip[q]);
            let Placement::Stream(from) = plan.nodes[p].placement else {
                continue;
            };
            match lane {
                None => {
                    let start = Instant::now();
                    let waited = backend.host_wait(NodeId(p), from_trip);
                    syncs.push(SyncPoint {
                        node: Some(NodeId(id)),
                        iteration,
                        worker: worker.to_string(),
                        waited_for: NodeId(p),
                        host: start.elapsed(),
                    });
                    waited?;
                }
                Some(to) if to != from => backend.wait(to, NodeId(p), from_trip)?,
                Some(_) => {}
            }
        }
        match (&node.work, lane) {
            (Work::Decide(l, decide), _) => {
                let lp = &plan.loops[*l];
                let LoopKind::Decide { max_iterations, .. } = lp.kind else {
                    unreachable!("a decision is its decision loop's")
                };
                for n in lp.body.clone() {
                    if plan.nodes[n].placement == Placement::Cpu {
                        continue;
                    }
                    let read = backend.elapsed(NodeId(n), 0).map_err(|e| {
                        e.with_context(format!("stream scope of {}", plan.nodes[n].name))
                    })?;
                    scopes.extend(read.map(|device| (n, device)));
                }
                let index = iteration.expect("a decision is in its loop");
                let start = Instant::now();
                let flow = decide(index);
                host = start.elapsed();
                if let ControlFlow::Continue(()) = flow? {
                    if index + 1 == max_iterations {
                        return Err(Error::max_iterations_exceeded(format!(
                            "the loop did not stop in {max_iterations} iterations"
                        )));
                    }
                    replay = true;
                }
                Ok(())
            }
            (Work::Task(run), None) => {
                let start = Instant::now();
                let result = run(Lane::Cpu {
                    iteration: iteration.unwrap_or(0),
                });
                host = start.elapsed();
                result
            }
            (Work::Task(run), Some(lane)) => {
                backend.begin(lane, NodeId(id), trip)?;
                let start = Instant::now();
                let result = run(Lane::Stream {
                    lane,
                    stream: backend.stream(lane),
                    iteration: iteration.unwrap_or(0),
                });
                host = start.elapsed();
                result?;
                backend.record(lane, NodeId(id), trip)
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
        iteration,
        host,
        device: None,
    };
    Ran {
        result: result.map_err(|e| e.with_context(format!("{}@{worker}", node.name))),
        time,
        syncs,
        replay,
        scopes,
    }
}
