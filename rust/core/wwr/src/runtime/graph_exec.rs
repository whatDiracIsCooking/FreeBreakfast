//! `wwr.extension.runtime:graph_exec`: the owning executable graph and the
//! borrow-safe launch operations.
//!
//! Not device-bound: an executable graph runs on the device of whatever
//! stream it is launched on.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_graph_exec, wwr_status, wwr_stream};

use super::graph::Graph;
use super::stream::Stream;
use crate::handle::{BaseHandle, DeviceHandleStream, HandleKind};
use fb_core_error_handling::check_or_abort;

/// An executable graph's destroy. It has no create of its own: an
/// executable is always instantiated from a graph, and the kind is not
/// exported, so [`GraphExec::new`] is the only way to make one.
pub struct GraphExecKind;

impl HandleKind for GraphExecKind {
    type Raw = wwr_graph_exec;
    type Error = wwr_status;

    /// Unreachable: no constructor takes the default create path.
    unsafe fn create(_: *mut wwr_graph_exec) -> wwr_status {
        wwr_status::WWR_STATUS_NOT_SUPPORTED
    }

    unsafe fn destroy(raw: wwr_graph_exec) -> wwr_status {
        // SAFETY: `raw` is a live executable graph, destroyed once.
        unsafe { wwr_sys::wwr_graph_exec_destroy(raw) }
    }
}

/// An owned executable graph: `GraphExecWrapper`.
pub struct GraphExec(BaseHandle<GraphExecKind>);

impl GraphExec {
    /// Instantiate `graph` (borrowed for this call only) with `flags`;
    /// aborts at the caller if it cannot.
    #[track_caller]
    pub fn new(graph: &Graph, flags: u64) -> Self {
        let graph = graph.get();
        GraphExec(check_or_abort(BaseHandle::try_new_at(
            // SAFETY: `raw` is valid for one write, and the borrow keeps the
            // graph alive for the call.
            |raw| unsafe { wwr_sys::wwr_graph_instantiate(raw, graph, flags) },
            Location::caller(),
        )))
    }

    /// [`launch`] this executable on `stream`.
    pub fn launch(&self, stream: &Stream) -> wwr_status {
        // SAFETY: both borrows keep their handles alive.
        unsafe { launch(self.get(), stream.stream()) }
    }

    /// [`upload`] this executable to `stream`'s device.
    pub fn upload(&self, stream: &Stream) -> wwr_status {
        // SAFETY: as `launch`.
        unsafe { upload(self.get(), stream.stream()) }
    }
}

impl Deref for GraphExec {
    type Target = BaseHandle<GraphExecKind>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// The borrow-safe operations on a raw executable graph, `unsafe` for the
// reason `stream`'s are.

/// Launch `exec` on `stream`.
///
/// # Safety
/// `exec` is a live executable graph, and `stream` a live stream or null
/// (the legacy default stream).
pub unsafe fn launch(exec: wwr_graph_exec, stream: wwr_stream) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_graph_launch(exec, stream) }
}

/// Upload `exec` to `stream`'s device without launching it.
///
/// # Safety
/// As [`launch`].
pub unsafe fn upload(exec: wwr_graph_exec, stream: wwr_stream) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_graph_upload(exec, stream) }
}
