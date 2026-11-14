//! A placed DAG: what [`run`](super::run) executes.

use std::collections::BTreeMap;

use fb_core_error_handling::Status;
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

pub(super) struct Node<'a, S> {
    pub(super) name: String,
    pub(super) placement: Placement,
    pub(super) preds: Vec<usize>,
    pub(super) dims: BTreeMap<String, usize>,
    pub(super) run: NodeFn<'a, S>,
}

/// A DAG of nodes, each placed, over the stream type `S` its GPU nodes
/// enqueue on (`()` under [`HostOnly`](super::HostOnly)).
///
/// A node's callable may borrow anything that outlives the plan (`'a`), and
/// must be `Sync`, because the run calls it from one of its workers.
pub struct Plan<'a, S> {
    pub(super) nodes: Vec<Node<'a, S>>,
    pub(super) succs: Vec<Vec<usize>>,
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
        }
    }

    /// Add a node after its predecessors `deps`, which must already be in
    /// the plan; that is what makes the plan acyclic and the ids a
    /// topological order.
    ///
    /// # Panics
    /// If a predecessor is not in this plan.
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
            assert!(p < id, "predecessor {p} is not in the plan");
            self.succs[p].push(id);
        }
        self.nodes.push(Node {
            name: name.into(),
            placement,
            preds,
            dims: BTreeMap::new(),
            run: Box::new(run),
        });
        self.succs.push(Vec::new());
        NodeId(id)
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
