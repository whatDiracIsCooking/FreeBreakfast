//! `wwr.extension.runtime:graph`: the owning graph.
//!
//! Not device-bound, in wwr or here: a graph's nodes may target different
//! devices, so there is no one device to record.

use std::ops::Deref;

use wwr_sys::{wwr_graph, wwr_status};

use super::graph_exec::GraphExec;
use crate::core::wwr::common::ErrorPolicy;
use crate::core::wwr::handle::{BaseHandle, HandleKind};

/// A graph's create (empty) and destroy.
pub struct GraphKind;

impl HandleKind for GraphKind {
    type Raw = wwr_graph;
    type Error = wwr_status;

    unsafe fn create(raw: *mut wwr_graph) -> wwr_status {
        // SAFETY: the caller hands a pointer valid for one write. The flags
        // word is reserved and must be 0.
        unsafe { wwr_sys::wwr_graph_create(raw, 0) }
    }

    unsafe fn destroy(raw: wwr_graph) -> wwr_status {
        // SAFETY: `raw` is a live graph, destroyed once.
        unsafe { wwr_sys::wwr_graph_destroy(raw) }
    }
}

/// An owned graph: `GraphWrapper<P_create, P_destroy>`. One comes empty
/// from [`Graph::new`], or recorded from a stream's
/// [`end_capture`](super::Stream::end_capture).
pub struct Graph<PC, PD>(pub(super) BaseHandle<GraphKind, PC, PD>)
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>;

impl<PC, PD> Graph<PC, PD>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
{
    /// An empty graph.
    #[track_caller]
    pub fn new(policy_create: PC, policy_destroy: PD) -> Self {
        Graph(BaseHandle::new(policy_create, policy_destroy))
    }

    /// Own `raw`, a graph made elsewhere: `GraphWrapper::adopt`. It is
    /// destroyed with this.
    ///
    /// # Safety
    /// `raw` is a live graph that nothing else destroys.
    pub unsafe fn adopt(raw: wwr_graph, policy_create: PC, policy_destroy: PD) -> Self {
        Graph(BaseHandle::new_with(
            policy_create,
            policy_destroy,
            |_, _| Some(raw),
        ))
    }

    /// Instantiate this graph for launching: `instantiate(flags)`. The
    /// executable reports through new defaults of this graph's policy
    /// types, as wwr default-constructs them.
    #[track_caller]
    pub fn instantiate(&self, flags: u64) -> GraphExec<PC, PD>
    where
        PC: Default,
        PD: Default,
    {
        GraphExec::new(self, flags, PC::default(), PD::default())
    }
}

impl<PC, PD> Deref for Graph<PC, PD>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
{
    type Target = BaseHandle<GraphKind, PC, PD>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wwr::common::tests::Recording;
    use crate::core::wwr::runtime::tests::Abort;

    /// Needs a GPU. An empty graph instantiates.
    #[test]
    fn an_empty_graph_instantiates() {
        let g = Graph::new(Abort::new(), Abort::new());
        assert!(g.get().is_some());
        assert!(g.instantiate(0).get().is_some());
    }

    /// Needs a GPU. Adopting hands the graph its destroy.
    #[test]
    fn adopt_owns_the_graph() {
        let mut raw: wwr_graph = std::ptr::null_mut();
        // SAFETY: writes one graph handle through a valid local.
        assert_eq!(
            unsafe { wwr_sys::wwr_graph_create(&mut raw, 0) },
            wwr_status::WWR_STATUS_SUCCESS
        );
        // SAFETY: `raw` is live and nothing else destroys it.
        let g = unsafe { Graph::adopt(raw, Recording::default(), Recording::default()) };
        assert_eq!(g.get(), Some(raw));
    }
}
