//! `wwr.extension.runtime:graph`: the owning graph.
//!
//! Not device-bound, in wwr or here: a graph's nodes may target different
//! devices, so there is no one device to record.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_graph, wwr_status};

use super::graph_exec::GraphExec;
use crate::handle::{BaseHandle, HandleKind};
use fb_core_error_handling::check_or_abort;

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

/// An owned graph: `GraphWrapper`. One comes empty from [`Graph::new`], or
/// recorded from a stream's [`end_capture`](super::Stream::end_capture).
pub struct Graph(BaseHandle<GraphKind>);

impl Graph {
    /// An empty graph; aborts at the caller if it cannot be made.
    #[track_caller]
    pub fn new() -> Self {
        Graph(check_or_abort(BaseHandle::try_new_at(
            // SAFETY: `try_new_at` hands a pointer valid for one write.
            |raw| unsafe { GraphKind::create(raw) },
            Location::caller(),
        )))
    }

    /// Own `raw`, a graph made elsewhere: `GraphWrapper::adopt`. It is
    /// destroyed with this.
    ///
    /// # Safety
    /// `raw` is a live graph that nothing else destroys.
    pub unsafe fn adopt(raw: wwr_graph) -> Self {
        // SAFETY: the caller's.
        Graph(unsafe { BaseHandle::adopt(raw) })
    }

    /// Instantiate this graph for launching: `instantiate(flags)`; aborts
    /// at the caller if it cannot.
    #[track_caller]
    pub fn instantiate(&self, flags: u64) -> GraphExec {
        GraphExec::new(self, flags)
    }
}

impl Default for Graph {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for Graph {
    type Target = BaseHandle<GraphKind>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a GPU. An empty graph instantiates.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn an_empty_graph_instantiates() {
        let g = Graph::new();
        let _exec = g.instantiate(0);
    }

    /// Needs a GPU. Adopting hands the graph its destroy.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn adopt_owns_the_graph() {
        let mut raw: wwr_graph = std::ptr::null_mut();
        // SAFETY: writes one graph handle through a valid local.
        assert_eq!(
            unsafe { wwr_sys::wwr_graph_create(&mut raw, 0) },
            wwr_status::WWR_STATUS_SUCCESS
        );
        // SAFETY: `raw` is live and nothing else destroys it.
        let g = unsafe { Graph::adopt(raw) };
        assert_eq!(g.get(), raw);
    }
}
