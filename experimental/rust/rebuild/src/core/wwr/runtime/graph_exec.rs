//! `wwr.extension.runtime:graph_exec`: the owning executable graph and the
//! borrow-safe launch operations.
//!
//! Not device-bound: an executable graph runs on the device of whatever
//! stream it is launched on.

use std::ops::Deref;

use wwr_sys::{wwr_graph_exec, wwr_status, wwr_stream};

use super::create_with;
use super::graph::Graph;
use super::stream::Stream;
use crate::core::wwr::common::ErrorPolicy;
use crate::core::wwr::handle::{BaseHandle, DeviceHandleStream, HandleKind};

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

/// An owned executable graph: `GraphExecWrapper<P_create, P_destroy>`.
pub struct GraphExec<PC, PD>(BaseHandle<GraphExecKind, PC, PD>)
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>;

impl<PC, PD> GraphExec<PC, PD>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
{
    /// Instantiate `graph` (borrowed for this call only) with `flags`.
    ///
    /// A graph that owns nothing is reported to the create policy as
    /// `WWR_STATUS_INVALID_VALUE` without reaching the runtime, where wwr
    /// hands the runtime its null.
    #[track_caller]
    pub fn new<GC, GD>(
        graph: &Graph<GC, GD>,
        flags: u64,
        policy_create: PC,
        policy_destroy: PD,
    ) -> Self
    where
        GC: ErrorPolicy<Error = wwr_status>,
        GD: ErrorPolicy<Error = wwr_status>,
    {
        let graph = graph.get();
        GraphExec(BaseHandle::new_with(
            policy_create,
            policy_destroy,
            |policy, location| {
                create_with(policy, location, |raw| match graph {
                    // SAFETY: `raw` is valid for one write, and the borrow
                    // keeps the graph alive for the call.
                    Some(graph) => unsafe { wwr_sys::wwr_graph_instantiate(raw, graph, flags) },
                    None => wwr_status::WWR_STATUS_INVALID_VALUE,
                })
            },
        ))
    }

    /// [`launch`] this executable on `stream`; `WWR_STATUS_INVALID_VALUE`
    /// if it owns nothing.
    pub fn launch<SC, SD, SDev>(&self, stream: &Stream<SC, SD, SDev>) -> wwr_status
    where
        SC: ErrorPolicy<Error = wwr_status>,
        SD: ErrorPolicy<Error = wwr_status>,
        SDev: ErrorPolicy<Error = wwr_status>,
    {
        match self.get() {
            // SAFETY: both borrows keep their handles alive.
            Some(raw) => unsafe { launch(raw, stream.stream()) },
            None => wwr_status::WWR_STATUS_INVALID_VALUE,
        }
    }

    /// [`upload`] this executable to `stream`'s device;
    /// `WWR_STATUS_INVALID_VALUE` if it owns nothing.
    pub fn upload<SC, SD, SDev>(&self, stream: &Stream<SC, SD, SDev>) -> wwr_status
    where
        SC: ErrorPolicy<Error = wwr_status>,
        SD: ErrorPolicy<Error = wwr_status>,
        SDev: ErrorPolicy<Error = wwr_status>,
    {
        match self.get() {
            // SAFETY: as `launch`.
            Some(raw) => unsafe { upload(raw, stream.stream()) },
            None => wwr_status::WWR_STATUS_INVALID_VALUE,
        }
    }
}

impl<PC, PD> Deref for GraphExec<PC, PD>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
{
    type Target = BaseHandle<GraphExecKind, PC, PD>;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wwr::common::tests::Recording;
    use crate::core::wwr::runtime::tests::Abort;

    /// Needs a GPU, for the stream. Instantiating a graph that owns nothing
    /// is reported through the create policy at the caller, the executable
    /// owns nothing, and launching it is refused.
    #[test]
    fn a_dead_graph_is_reported_and_its_executable_refused() {
        // Ending a capture that never began leaves the graph empty-handed.
        let mut s = Stream::new(0, Recording::default(), Recording::default(), Abort::new());
        let dead = s.end_capture();
        assert_eq!(dead.get(), None);

        let line = line!() + 1;
        let exec = GraphExec::new(&dead, 0, Recording::default(), Recording::default());
        assert_eq!(exec.get(), None);
        assert_eq!(
            exec.create_policy().0,
            [(wwr_status::WWR_STATUS_INVALID_VALUE, line)]
        );
        assert_eq!(exec.launch(&s), wwr_status::WWR_STATUS_INVALID_VALUE);
        assert_eq!(exec.upload(&s), wwr_status::WWR_STATUS_INVALID_VALUE);
    }
}
