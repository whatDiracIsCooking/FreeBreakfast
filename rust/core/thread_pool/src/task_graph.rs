//! A placed DAG: what [`run`](super::run) executes.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{ControlFlow, Range};

use contracts::debug_requires;
use fb_core_error_handling::{Result, Status};
use serde::{Deserialize, Serialize};

/// Where a node runs: the CPU pool, or one lane of one GPU worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Any of the run's CPU workers.
    Cpu,
    /// A lane: enqueued by its GPU worker's thread, on its device, on the
    /// lane's stream.
    Stream(LaneId),
}

/// One stream of one GPU worker: lane `lane` of worker `worker`
/// (`docs/programming_model.md`, "Workers are a hierarchy").
///
/// A GPU worker is one card, one thread and several streams, its lanes. Its
/// thread enqueues its ready nodes in rank order across all its lanes and
/// never waits on the device; two nodes on one lane are ordered by the
/// stream, and an edge between two lanes is an event one records and the
/// other waits on, on the device. A worker given as a bare index is its
/// lane 0, so a graph of one lane per worker names none.
///
/// Its text, `gpu<worker>.<lane>`, is the worker a node on it is logged
/// and timed under (`[name@gpu0.1]`).
///
/// ```
/// use fb_core_thread_pool::{HostOnly, LaneId, Placement, TaskGraph, run};
///
/// let lane = |lane| Placement::Stream(LaneId { worker: 0, lane });
/// let mut plan = TaskGraph::new();
/// let a = plan.add("a", lane(0), &[], |_| Ok(()));
/// plan.add("b", lane(1), &[a], |_| Ok(()));
/// assert_eq!(plan.lanes(), [2]);
/// let record = run(&mut plan, &mut HostOnly::new(1), 1);
/// record.result.unwrap();
/// assert_eq!(record.tasks[1].worker, "gpu0.1");
/// // b's lane waits for a on the device: the only syncs are the lanes' exits.
/// assert!(record.syncs.iter().all(|s| s.node.is_none()));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneId {
    /// The GPU worker: its card, its thread and its blocks.
    pub worker: usize,
    /// The lane among the worker's: its stream.
    pub lane: usize,
}

impl From<usize> for LaneId {
    /// Worker `worker`'s lane 0.
    fn from(worker: usize) -> Self {
        LaneId { worker, lane: 0 }
    }
}

impl fmt::Display for LaneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu{}.{}", self.worker, self.lane)
    }
}

/// A node in a [`TaskGraph`], and its rank: nodes are numbered in the order they
/// were added, which is a topological order. In `timing.json` it is the
/// rank, a bare integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub(super) usize);

impl NodeId {
    /// The node's rank, its index in the plan.
    pub fn index(self) -> usize {
        self.0
    }
}

/// What a node is handed when it runs: where, and in which iteration of
/// the innermost loop it is in, from 0, or 0 outside every loop. A counted
/// loop's body node picks its batch by it ([`TaskGraph::add_counted`]), inside
/// a decision loop too: there it is the trip, and the decision loop's
/// iteration is the record's ([`TaskTime::outer`](super::TaskTime::outer)).
pub enum Lane<'r, S> {
    /// A CPU worker: compute on the host.
    Cpu {
        /// The iteration.
        iteration: usize,
    },
    /// A lane of a GPU worker: enqueue on `stream` and return. The worker's
    /// device is current.
    Stream {
        /// The lane.
        lane: LaneId,
        /// Its stream; under the `Device` backend, the
        /// [`DeviceStream`](super::DeviceStream) that also carries the
        /// lane's library handles.
        stream: &'r S,
        /// The iteration.
        iteration: usize,
    },
}

type NodeFn<'a, S> = Box<dyn Fn(Lane<'_, S>) -> Status + Sync + 'a>;
type DecideFn<'a> = Box<dyn Fn(usize) -> Result<ControlFlow<()>> + Sync + 'a>;

/// What a node runs: a task, or a loop's decision.
pub(super) enum Work<'a, S> {
    Task(NodeFn<'a, S>),
    /// The decision of loop `.0` in [`TaskGraph::loops`].
    Decide(usize, DecideFn<'a>),
}

pub(super) struct Node<'a, S> {
    pub(super) name: String,
    pub(super) placement: Placement,
    pub(super) preds: Vec<usize>,
    pub(super) dims: BTreeMap<String, usize>,
    /// The innermost loop whose body or decision it is.
    pub(super) in_loop: Option<usize>,
    pub(super) work: Work<'a, S>,
}

/// A loop of a [`TaskGraph`]: its body's ranks, how it ends, and the decision
/// loop whose body holds it, if any.
pub(super) struct Loop {
    pub(super) body: Range<usize>,
    pub(super) kind: LoopKind,
    pub(super) parent: Option<usize>,
}

/// How a loop ends: at a decision, or after a trip count.
#[derive(Clone, Copy)]
pub(super) enum LoopKind {
    /// [`TaskGraph::add_loop`]'s: its decision's rank and its bound.
    Decide {
        decision: usize,
        max_iterations: usize,
    },
    /// [`TaskGraph::add_counted`]'s: its trip count and its depth.
    Counted { trips: usize, depth: usize },
}

impl Loop {
    /// Its last rank: its decision's, or its body's last.
    pub(super) fn last(&self) -> usize {
        match self.kind {
            LoopKind::Decide { decision, .. } => decision,
            LoopKind::Counted { .. } => self.body.end - 1,
        }
    }

    /// Its decision's rank, for a loop that has one.
    pub(super) fn decision(&self) -> Option<usize> {
        match self.kind {
            LoopKind::Decide { decision, .. } => Some(decision),
            LoopKind::Counted { .. } => None,
        }
    }
}

/// A DAG of nodes, each placed, over the stream type `S` its GPU nodes
/// enqueue on (`()` under [`HostOnly`](super::HostOnly)), and the loops
/// that replay a run of them ([`TaskGraph::add_loop`]).
///
/// A node's callable may borrow anything that outlives the graph (`'a`), and
/// must be `Sync`, because the run calls it from one of its workers.
pub struct TaskGraph<'a, S> {
    pub(super) nodes: Vec<Node<'a, S>>,
    pub(super) succs: Vec<Vec<usize>>,
    pub(super) loops: Vec<Loop>,
}

impl<S> Default for TaskGraph<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, S> TaskGraph<'a, S> {
    /// An empty plan.
    pub fn new() -> Self {
        TaskGraph {
            nodes: Vec::new(),
            succs: Vec::new(),
            loops: Vec::new(),
        }
    }

    /// Add a node after its predecessors `deps`, which must already be in
    /// the graph; that is what makes the graph acyclic and the ids a
    /// topological order. A node after a decision loop reads what the
    /// loop's body wrote through the loop's decision, never from a body
    /// node, which runs once per iteration. A node after a counted loop
    /// may follow a body node: it runs after that node's last iteration.
    #[debug_requires(
        deps.iter().all(|d| d.0 < self.nodes.len()),
        "every predecessor is in the plan"
    )]
    #[debug_requires(
        deps.iter().all(|d| self.loops.iter().all(|l| {
            l.decision().is_none() || !l.body.contains(&d.0)
        })),
        "no predecessor is in a closed loop's body"
    )]
    pub fn add(
        &mut self,
        name: impl Into<String>,
        placement: Placement,
        deps: &[NodeId],
        run: impl Fn(Lane<'_, S>) -> Status + Sync + 'a,
    ) -> NodeId {
        let id = self.nodes.len();
        let mut preds: Vec<usize> = deps.iter().map(|d| d.0).collect();
        preds.sort_unstable();
        preds.dedup();
        for &p in &preds {
            self.succs[p].push(id);
        }
        self.nodes.push(Node {
            name: name.into(),
            placement,
            preds,
            dims: BTreeMap::new(),
            in_loop: None,
            work: Work::Task(Box::new(run)),
        });
        self.succs.push(Vec::new());
        NodeId(id)
    }

    /// Close a loop: the nodes from `body` to the last one added are its
    /// body, a sub-DAG the run replays on the same workers until `decide`
    /// stops it (`docs/programming_model.md`, "Plan once, run many").
    /// Returns the decision node, which is what a node after the loop
    /// depends on to read the loop's outcome.
    ///
    /// The decision is a CPU node after the body's sinks, so after every
    /// node of the body, and after `deps`, which may name nodes before the
    /// body that no body node follows (a row the decision reads, or whose
    /// bytes it takes over); a body node in `deps` adds no edge and no host
    /// wait, the sinks being after it already. It runs once per iteration, when the whole body has finished and every
    /// GPU body node's work has completed, and is handed the iteration's
    /// index from 0. Its `Continue` readies the body again, inside the same run, with
    /// nothing spawned and nothing allocated by the executor; its `Break`
    /// readies the nodes after the loop. A body node's predecessors before
    /// the body are satisfied once, by their single run.
    ///
    /// The body may hold counted loops ([`TaskGraph::add_counted`]), closed
    /// before this one: each is replayed whole every iteration, its trips
    /// overlapped as anywhere, and the decision reads each trip's stream
    /// scopes before the replay.
    ///
    /// The body runs at most `max_iterations` times, a bound fixed here, at
    /// planning: a decision that still says `Continue` at index
    /// `max_iterations - 1` fails as
    /// [`MaxIterationsExceeded`](fb_core_error_handling::ErrorKind::MaxIterationsExceeded),
    /// and a failed body node or decision ends the loop and skips the nodes
    /// after it, as any failure skips its descendants.
    ///
    /// ```
    /// use std::ops::ControlFlow;
    /// use std::sync::Mutex;
    /// use fb_core_thread_pool::{HostOnly, Placement, TaskGraph, run};
    ///
    /// // x <- x / 2 + 1 from 0 until it moves 1e-12 or less: the k-th step
    /// // moves 2^(1-k), so the 41st is the first to stop, 2^-40 from 2.
    /// let x = Mutex::new((0.0_f64, f64::INFINITY));
    /// let mut plan = TaskGraph::new();
    /// let step = plan.add("step", Placement::Cpu, &[], |_| {
    ///     let mut x = x.lock().unwrap();
    ///     let next = x.0 / 2.0 + 1.0;
    ///     *x = (next, (next - x.0).abs());
    ///     Ok(())
    /// });
    /// let done = plan.add_loop("converged", step, &[], 100, |_iteration| {
    ///     let moved = x.lock().unwrap().1;
    ///     Ok(if moved > 1e-12 { ControlFlow::Continue(()) } else { ControlFlow::Break(()) })
    /// });
    /// plan.add("report", Placement::Cpu, &[done], |_| Ok(()));
    /// let record = run(&mut plan, &mut HostOnly::new(0), 1);
    /// record.result.unwrap();
    /// assert!((x.lock().unwrap().0 - 2.0).abs() <= 1e-12);
    /// assert_eq!(record.iterations[&done], 41);
    /// ```
    #[debug_requires(body.0 < self.nodes.len(), "the body has a node")]
    #[debug_requires(
        self.loops.iter().all(|l| l.last() < body.0 || l.decision().is_none()),
        "a decision loop holds no decision loop"
    )]
    #[debug_requires(
        self.loops.iter().all(|l| l.last() < body.0 || l.body.start >= body.0),
        "a loop holds another whole or not at all"
    )]
    #[debug_requires(max_iterations > 0, "the bound admits an iteration")]
    #[debug_requires(
        deps.iter().all(|d| d.0 < self.nodes.len()),
        "every predecessor is in the plan"
    )]
    pub fn add_loop(
        &mut self,
        name: impl Into<String>,
        body: NodeId,
        deps: &[NodeId],
        max_iterations: usize,
        decide: impl Fn(usize) -> Result<ControlFlow<()>> + Sync + 'a,
    ) -> NodeId {
        let body = body.0..self.nodes.len();
        let index = self.loops.len();
        // The body's sinks: every body node is one or an ancestor of one,
        // so after them is after the whole body, and a body node given in
        // `deps` adds nothing, not even a host wait.
        let mut preds: Vec<NodeId> = body
            .clone()
            .filter(|&n| self.succs[n].is_empty())
            .map(NodeId)
            .collect();
        preds.extend(deps.iter().filter(|d| !body.contains(&d.0)));
        let decision = self.add(name, Placement::Cpu, &preds, |_| Ok(()));
        self.nodes[decision.0].work = Work::Decide(index, Box::new(decide));
        // A counted loop in the body keeps its nodes, and is held by this one.
        for n in body.start..=decision.0 {
            self.nodes[n].in_loop.get_or_insert(index);
        }
        for l in self.loops.iter_mut().filter(|l| l.body.start >= body.start) {
            l.parent = Some(index);
        }
        self.loops.push(Loop {
            body,
            kind: LoopKind::Decide {
                decision: decision.0,
                max_iterations,
            },
            parent: None,
        });
        decision
    }

    /// Close a counted loop: the nodes from `body` to the last one added
    /// are its body, run `trips` times in the same run, with no decision
    /// and no host wait between iterations (`docs/programming_model.md`,
    /// "Plan once, run many"). Each body node is handed its iteration's
    /// index ([`Lane`]), by which it picks its batch.
    ///
    /// Iterations overlap: iteration `k` waits on iteration `k - depth`
    /// alone, each body node with no predecessor in the body after every
    /// body node with no successor there, the ring that lets a planner
    /// hold `depth` copies of the body's rows. Each body node also runs
    /// after itself in the iteration before, which on a lane is the
    /// stream's order anyway. Each edge is satisfied as any edge is ([`thread_pool`](super),
    /// "When an edge is satisfied"), so between two lanes the wait is on
    /// the device and no GPU node of the loop adds a sync point: a GPU
    /// worker enqueues every iteration back to back, and a CPU body node
    /// waits on the host only for what it reads. A body node's
    /// predecessors before the body are satisfied once, by their single
    /// run.
    ///
    /// A node added after the loop may follow a body node, and then runs
    /// after its last iteration, so after all of them. A failed body node
    /// skips its descendants, every later iteration among them.
    ///
    /// A counted loop may lie in a decision loop's body, closed before the
    /// decision ([`TaskGraph::add_loop`]); it holds no loop itself.
    ///
    /// ```
    /// use std::sync::Mutex;
    /// use fb_core_thread_pool::{HostOnly, Lane, LaneId, Placement, TaskGraph, run};
    ///
    /// // A batch per iteration: lane 1 stages it, lane 0 adds it in.
    /// let staged = Mutex::new([0; 2]);
    /// let sum = Mutex::new(0);
    /// let lane = |lane| Placement::Stream(LaneId { worker: 0, lane });
    /// let mut plan = TaskGraph::new();
    /// let stage = plan.add("stage", lane(1), &[], |at| {
    ///     let Lane::Stream { iteration, .. } = at else { unreachable!() };
    ///     staged.lock().unwrap()[iteration % 2] = iteration + 1;
    ///     Ok(())
    /// });
    /// let add = plan.add("add", lane(0), &[stage], |at| {
    ///     let Lane::Stream { iteration, .. } = at else { unreachable!() };
    ///     *sum.lock().unwrap() += staged.lock().unwrap()[iteration % 2];
    ///     Ok(())
    /// });
    /// plan.add_counted(stage, 8, 2);
    /// plan.add("report", Placement::Cpu, &[add], |_| Ok(()));
    /// let record = run(&mut plan, &mut HostOnly::new(1), 1);
    /// record.result.unwrap();
    /// assert_eq!(*sum.lock().unwrap(), (1..=8).sum::<usize>());
    /// assert_eq!(record.tasks.iter().filter(|t| t.name == "add").count(), 8);
    /// ```
    #[debug_requires(body.0 < self.nodes.len(), "the body has a node")]
    #[debug_requires(
        self.loops.last().is_none_or(|l| l.last() < body.0),
        "a counted loop holds no loop"
    )]
    #[debug_requires(trips > 0, "the loop has an iteration")]
    #[debug_requires(0 < depth && depth <= trips, "the depth is from 1 to the trip count")]
    pub fn add_counted(&mut self, body: NodeId, trips: usize, depth: usize) {
        let body = body.0..self.nodes.len();
        let index = self.loops.len();
        for n in body.clone() {
            self.nodes[n].in_loop = Some(index);
        }
        self.loops.push(Loop {
            body,
            kind: LoopKind::Counted { trips, depth },
            parent: None,
        });
    }

    /// Name `node`'s dimensions, each by its name, for its entry in the
    /// timing record ([`TaskTime::dims`](super::TaskTime::dims)): what the
    /// nightly keys a task's times by beside its name. A name given twice
    /// keeps its last value.
    pub fn set_dims(&mut self, node: NodeId, dims: &[(&str, usize)]) {
        let to = &mut self.nodes[node.0].dims;
        to.extend(dims.iter().map(|&(name, n)| (name.to_string(), n)));
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Where `node` runs.
    pub fn placement(&self, node: NodeId) -> Placement {
        self.nodes[node.0].placement
    }

    /// `node`'s name.
    pub fn name(&self, node: NodeId) -> &str {
        &self.nodes[node.0].name
    }

    /// How many trips `node` runs per run of its loop: its counted loop's
    /// trip count, else 1.
    pub(super) fn trips(&self, node: &Node<'_, S>) -> usize {
        match node.in_loop.map(|l| self.loops[l].kind) {
            Some(LoopKind::Counted { trips, .. }) => trips,
            _ => 1,
        }
    }

    /// The lanes the graph uses, per GPU worker: for each worker up to the
    /// highest placed on, one past the highest lane placed on it, 0 for a
    /// worker with no node of its own.
    pub fn lanes(&self) -> Vec<usize> {
        let mut lanes = Vec::new();
        for n in &self.nodes {
            if let Placement::Stream(at) = n.placement {
                if lanes.len() <= at.worker {
                    lanes.resize(at.worker + 1, 0);
                }
                lanes[at.worker] = lanes[at.worker].max(at.lane + 1);
            }
        }
        lanes
    }
}
