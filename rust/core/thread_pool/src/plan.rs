//! A placed DAG: what [`run`](super::run) executes.

use std::collections::BTreeMap;
use std::ops::{ControlFlow, Range};

use contracts::debug_requires;
use fb_core_error_handling::{Result, Status};
use serde::{Deserialize, Serialize};

/// Where a node runs: the CPU pool, or one stream worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Any of the run's CPU workers.
    Cpu,
    /// Stream worker `s`: its thread, its device and its stream.
    Stream(usize),
}

/// A node in a [`Plan`], and its rank: nodes are numbered in the order they
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

/// What a node is handed when it runs.
pub enum Lane<'r, S> {
    /// A CPU worker: compute on the host.
    Cpu,
    /// Stream worker `index`: enqueue on `stream` and return. The worker's
    /// device is current.
    Stream {
        /// The stream worker's index.
        index: usize,
        /// Its stream; under the `Device` backend, the
        /// [`DeviceStream`](super::DeviceStream) that also carries the
        /// worker's library handles.
        stream: &'r S,
    },
}

type NodeFn<'a, S> = Box<dyn Fn(Lane<'_, S>) -> Status + Sync + 'a>;
type DecideFn<'a> = Box<dyn Fn(usize) -> Result<ControlFlow<()>> + Sync + 'a>;

/// What a node runs: a task, or a loop's decision.
pub(super) enum Work<'a, S> {
    Task(NodeFn<'a, S>),
    /// The decision of loop `.0` in [`Plan::loops`].
    Decide(usize, DecideFn<'a>),
}

pub(super) struct Node<'a, S> {
    pub(super) name: String,
    pub(super) placement: Placement,
    pub(super) preds: Vec<usize>,
    pub(super) dims: BTreeMap<String, usize>,
    /// The loop whose body or decision it is.
    pub(super) in_loop: Option<usize>,
    pub(super) work: Work<'a, S>,
}

/// A loop of a [`Plan`]: its body's ranks, its decision's, and its bound.
pub(super) struct Loop {
    pub(super) body: Range<usize>,
    pub(super) decision: usize,
    pub(super) max_iterations: usize,
}

/// A DAG of nodes, each placed, over the stream type `S` its GPU nodes
/// enqueue on (`()` under [`HostOnly`](super::HostOnly)), and the loops
/// that replay a run of them ([`Plan::add_loop`]).
///
/// A node's callable may borrow anything that outlives the plan (`'a`), and
/// must be `Sync`, because the run calls it from one of its workers.
pub struct Plan<'a, S> {
    pub(super) nodes: Vec<Node<'a, S>>,
    pub(super) succs: Vec<Vec<usize>>,
    pub(super) loops: Vec<Loop>,
}

impl<S> Default for Plan<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, S> Plan<'a, S> {
    /// An empty plan.
    pub fn new() -> Self {
        Plan {
            nodes: Vec::new(),
            succs: Vec::new(),
            loops: Vec::new(),
        }
    }

    /// Add a node after its predecessors `deps`, which must already be in
    /// the plan; that is what makes the plan acyclic and the ids a
    /// topological order. A node after a loop reads what the loop's body
    /// wrote through the loop's decision, never from a body node, which
    /// runs once per iteration.
    #[debug_requires(
        deps.iter().all(|d| d.0 < self.nodes.len()),
        "every predecessor is in the plan"
    )]
    #[debug_requires(
        deps.iter().all(|d| self.loops.iter().all(|l| !l.body.contains(&d.0))),
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
    /// The decision is a CPU node after every node of the body and after
    /// `deps`, which may name nodes before the body that no body node
    /// follows (a row the decision reads, or whose bytes it takes over): it
    /// runs once per iteration, when the whole body has finished and every
    /// GPU body node's work has completed, and is handed the iteration's
    /// index from 0. Its `Continue` readies the body again, inside the same run, with
    /// nothing spawned and nothing allocated by the executor; its `Break`
    /// readies the nodes after the loop. A body node's predecessors before
    /// the body are satisfied once, by their single run.
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
    /// use fb_core_thread_pool::{HostOnly, Placement, Plan, run};
    ///
    /// // x <- x / 2 + 1 from 0 until it moves 1e-12 or less: the k-th step
    /// // moves 2^(1-k), so the 41st is the first to stop, 2^-40 from 2.
    /// let x = Mutex::new((0.0_f64, f64::INFINITY));
    /// let mut plan = Plan::new();
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
        self.loops.last().is_none_or(|l| l.decision < body.0),
        "a body holds no loop"
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
        // so after them is after the whole body.
        let mut preds: Vec<NodeId> = body
            .clone()
            .filter(|&n| self.succs[n].is_empty())
            .map(NodeId)
            .collect();
        preds.extend_from_slice(deps);
        let decision = self.add(name, Placement::Cpu, &preds, |_| Ok(()));
        self.nodes[decision.0].work = Work::Decide(index, Box::new(decide));
        for n in body.start..=decision.0 {
            self.nodes[n].in_loop = Some(index);
        }
        self.loops.push(Loop {
            body,
            decision: decision.0,
            max_iterations,
        });
        decision
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

    /// Whether the plan has no nodes.
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

    /// The stream workers the plan uses: one past the highest index placed.
    pub fn n_streams(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| match n.placement {
                Placement::Stream(s) => Some(s + 1),
                Placement::Cpu => None,
            })
            .max()
            .unwrap_or(0)
    }
}
