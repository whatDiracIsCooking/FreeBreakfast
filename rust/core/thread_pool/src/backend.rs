//! The device side of a run.

use std::time::Duration;

use super::plan::{LaneId, NodeId};
#[cfg(feature = "device")]
use super::plan::{Placement, Plan};
#[cfg(feature = "device")]
use contracts::debug_requires;
#[cfg(feature = "device")]
use fb_core_error_handling::Error;
use fb_core_error_handling::{Result, Status};
#[cfg(feature = "device")]
use fb_core_wwr::blas::BlasHandle;
#[cfg(feature = "device")]
use fb_core_wwr::common::ScopedDeviceIndex;
#[cfg(feature = "device")]
use fb_core_wwr::device_slice::DeviceSliceMut;
#[cfg(feature = "device")]
use fb_core_wwr::gpu_status;
#[cfg(feature = "device")]
use fb_core_wwr::runtime::{Event, Stream, event};
#[cfg(feature = "device")]
use fb_core_wwr::solver::SolverHandle;
#[cfg(feature = "device")]
use std::ops::Deref;
#[cfg(feature = "device")]
use std::rc::Rc;

/// The device side of a run: the lanes the GPU workers enqueue on, and the
/// waits that make each edge leaving a lane hold
/// ([`thread_pool`](super), "When an edge is satisfied").
///
/// Every method is called from the worker the comment names.
/// [`run`](super::run) borrows the backend exclusively and shares it with
/// its workers through a wrapper only it can make, so a backend is on
/// several threads only inside one run, and never in two runs at once. A
/// backend need not be `Sync`, and `Device` is not: outside a run no
/// `&Device` can reach another thread at all.
///
/// Sealed: [`HostOnly`] and `Device` are the implementations, so `run`'s
/// argument for sharing one across its workers covers every backend.
pub trait Backend: sealed::Sealed {
    /// What a GPU node enqueues on: a lane's stream.
    type Stream;
    /// Held by a GPU worker's thread for the whole run: its device made
    /// current. Made and dropped on that thread.
    type Binding;

    /// The number of GPU workers, each one thread of the run.
    fn n_workers(&self) -> usize;
    /// Whether the backend has `lane`, so a node may be placed on it.
    fn has_lane(&self, lane: LaneId) -> bool;
    /// On GPU worker `worker`'s thread, as it starts.
    fn bind(&self, worker: usize) -> Self::Binding;
    /// `lane`'s stream.
    fn stream(&self, lane: LaneId) -> &Self::Stream;
    /// On `lane`'s worker, just before GPU node `node` runs on it: open its
    /// stream scope.
    fn begin(&self, lane: LaneId, node: NodeId) -> Status;
    /// On `lane`'s worker, after GPU node `node` returned: mark the point
    /// its consumers wait for, which also closes its stream scope.
    fn record(&self, lane: LaneId, node: NodeId) -> Status;
    /// On `lane`'s worker, before a node on `lane` that consumes `node`,
    /// which ran on another lane: make `lane` wait for it on the device,
    /// with no host wait.
    fn wait(&self, lane: LaneId, node: NodeId) -> Status;
    /// On a CPU worker, before a node that consumes GPU node `node`: wait
    /// for it on the host.
    fn host_wait(&self, node: NodeId) -> Status;
    /// On `lane`'s worker, as it exits: wait for all the lane's work.
    fn sync(&self, lane: LaneId) -> Status;
    /// After the run, every stream synced: the device time of GPU node
    /// `node`'s stream scope, which [`begin`](Backend::begin) and
    /// [`record`](Backend::record) both succeeded on in this run; `None`
    /// for a backend with no device clock.
    fn elapsed(&self, node: NodeId) -> Result<Option<Duration>>;
}

/// Nothing outside this module can name it, so nothing outside the crate
/// can implement [`Backend`].
mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for HostOnly {}
#[cfg(feature = "device")]
impl sealed::Sealed for Device {}

/// No device: `n` GPU workers that are plain threads, whose nodes are
/// complete when they return. A lane here is a label with no stream behind
/// it, so each worker has every lane a plan names. For device-free tests of
/// the executor and of the DAGs built on it.
#[derive(Clone, Copy, Debug)]
pub struct HostOnly {
    n_workers: usize,
}

impl HostOnly {
    /// `n_workers` GPU workers.
    pub fn new(n_workers: usize) -> Self {
        HostOnly { n_workers }
    }
}

impl Backend for HostOnly {
    type Stream = ();
    type Binding = ();

    fn n_workers(&self) -> usize {
        self.n_workers
    }
    fn has_lane(&self, lane: LaneId) -> bool {
        lane.worker < self.n_workers
    }
    fn bind(&self, _: usize) {}
    fn stream(&self, _: LaneId) -> &() {
        &()
    }
    fn begin(&self, _: LaneId, _: NodeId) -> Status {
        Ok(())
    }
    fn record(&self, _: LaneId, _: NodeId) -> Status {
        Ok(())
    }
    fn wait(&self, _: LaneId, _: NodeId) -> Status {
        Ok(())
    }
    fn host_wait(&self, _: NodeId) -> Status {
        Ok(())
    }
    fn sync(&self, _: LaneId) -> Status {
        Ok(())
    }
    fn elapsed(&self, _: NodeId) -> Result<Option<Duration>> {
        Ok(None)
    }
}

/// What a GPU node is handed under [`Device`]: its lane's stream, which it
/// derefs to, and the lane's library handles, each bound to that stream.
/// Made with the [`Device`], never during a run, so a task reaches a handle
/// through its [`Lane`](super::Lane) and never makes one.
///
/// A handle per lane, not per worker: two lanes run at once on the card,
/// and a handle's state and its workspace are one stream's at a time. Every
/// GPU node of the lane shares its handles, so a node that changes a
/// handle's state (its pointer mode, say) sets it back before it returns;
/// none does today. The libraries are BLAS and the dense solver. The BLAS
/// handle's workspace is the backend's to hand over
/// ([`Device::bind_blas_workspace`]), never a node's.
#[cfg(feature = "device")]
pub struct DeviceStream {
    // Fields drop in declaration order: the handles before the stream they
    // are bound to (each also holds the stream until it is destroyed).
    blas: BlasHandle,
    solver: SolverHandle,
    stream: Rc<Stream>,
}

#[cfg(feature = "device")]
impl DeviceStream {
    /// A non-blocking stream on `dev_idx`, and a BLAS and a dense-solver
    /// handle bound to it. Aborts if any cannot be made.
    #[track_caller]
    pub fn new(dev_idx: i32) -> Self {
        let stream = Rc::new(Stream::new(dev_idx));
        DeviceStream {
            blas: BlasHandle::new(Rc::clone(&stream)),
            solver: SolverHandle::new(Rc::clone(&stream)),
            stream,
        }
    }

    /// The lane's BLAS handle, bound to its stream: pass its `get()` to
    /// a `wwr_blas_*` call, and the call enqueues on the stream.
    pub fn blas(&self) -> &BlasHandle {
        &self.blas
    }

    /// The lane's dense-solver handle, bound to its stream: pass its
    /// `get()` to a `wwr_solver_dn_*` call, and the call enqueues on the
    /// stream.
    pub fn solver(&self) -> &SolverHandle {
        &self.solver
    }
}

#[cfg(feature = "device")]
impl Deref for DeviceStream {
    type Target = Stream;
    fn deref(&self) -> &Stream {
        &self.stream
    }
}

/// Without the `device` feature: a stream that cannot exist, so the
/// planner's production default (`GpuTask<Pinned, DeviceStream>`) still names a
/// type, and a task over it can be declared and planned but never run,
/// since there is no `Device` backend to run it on.
#[cfg(not(feature = "device"))]
#[derive(Debug)]
pub enum DeviceStream {}

#[cfg(feature = "device")]
/// GPU workers, each on its own device with one non-blocking stream per
/// lane and the lane's library handles bound to it ([`DeviceStream`]), and
/// two timing events per GPU node: its stream scope, recorded before it
/// runs and after it returns. The second is also what its consumers wait
/// on.
///
/// Built once per plan and reused every run, one run at a time
/// ([`run`](super::run) borrows it exclusively): the streams, handles and
/// events are created here, never during a run.
///
/// Neither `Send` nor `Sync`, as the raw runtime handles inside it are not:
/// only a run shares it across threads, its workers, through `run`'s own
/// wrapper. So no code outside a run can call through its handles from
/// two threads:
///
/// ```compile_fail,E0277
/// fn shared_across_threads<T: Sync>() {}
/// shared_across_threads::<fb_core_thread_pool::Device>();
/// ```
pub struct Device {
    devices: Vec<i32>,
    /// Per worker, its lanes' streams.
    streams: Vec<Vec<DeviceStream>>,
    /// Indexed by node, each GPU node's (start, end); `None` for a CPU node.
    events: Vec<Option<(Event, Event)>>,
}

#[cfg(feature = "device")]
impl Device {
    /// GPU worker `w` on device `devices[w]`, with as many lanes as `plan`
    /// places nodes on ([`Plan::lanes`]), at least one, and an event pair
    /// for every GPU node of `plan`.
    #[debug_requires(
        plan.lanes().len() <= devices.len(),
        "a device for every GPU worker of the plan"
    )]
    #[track_caller]
    pub fn new(devices: &[i32], plan: &Plan<'_, DeviceStream>) -> Self {
        let lanes = plan.lanes();
        let streams = devices
            .iter()
            .enumerate()
            .map(|(w, &d)| {
                let n = lanes.get(w).copied().unwrap_or(0).max(1);
                (0..n).map(|_| DeviceStream::new(d)).collect()
            })
            .collect();
        let events = plan
            .nodes
            .iter()
            .map(|n| match n.placement {
                // Timing-capable: `DISABLE_TIMING` left off.
                Placement::Stream(at) => Some((
                    Event::with_flags(devices[at.worker], event::BLOCKING_SYNC),
                    Event::with_flags(devices[at.worker], event::BLOCKING_SYNC),
                )),
                Placement::Cpu => None,
            })
            .collect();
        Device {
            devices: devices.to_vec(),
            streams,
            events,
        }
    }

    /// Hand `lane`'s BLAS handle `workspace`, memory on the lane's device,
    /// which the handle owns from here until the backend drops or hands it
    /// another ([`BlasHandle::bind_workspace`]). `&mut self`, so never
    /// during a run: a planner instance binds each lane's declared row when
    /// it makes its backend.
    ///
    /// # Errors
    /// As [`BlasHandle::bind_workspace`]: the library's refusal.
    ///
    /// # Panics
    /// If the backend does not have `lane`.
    pub fn bind_blas_workspace<W>(&mut self, lane: LaneId, workspace: W) -> Status
    where
        W: DeviceSliceMut<u8> + Send + Sync + 'static,
    {
        self.streams[lane.worker][lane.lane]
            .blas
            .bind_workspace(workspace)
    }

    fn events(&self, node: NodeId) -> &(Event, Event) {
        self.events[node.index()]
            .as_ref()
            .expect("an event is recorded only for a GPU node")
    }
}

#[cfg(feature = "device")]
impl Backend for Device {
    type Stream = DeviceStream;
    type Binding = ScopedDeviceIndex;

    fn n_workers(&self) -> usize {
        self.streams.len()
    }
    fn has_lane(&self, lane: LaneId) -> bool {
        self.streams
            .get(lane.worker)
            .is_some_and(|lanes| lane.lane < lanes.len())
    }
    fn bind(&self, worker: usize) -> Self::Binding {
        ScopedDeviceIndex::new(self.devices[worker])
    }
    fn stream(&self, lane: LaneId) -> &DeviceStream {
        &self.streams[lane.worker][lane.lane]
    }
    fn begin(&self, lane: LaneId, node: NodeId) -> Status {
        gpu_status(self.events(node).0.record(self.stream(lane)))
    }
    fn record(&self, lane: LaneId, node: NodeId) -> Status {
        gpu_status(self.events(node).1.record(self.stream(lane)))
    }
    fn wait(&self, lane: LaneId, node: NodeId) -> Status {
        gpu_status(self.stream(lane).wait_event(&self.events(node).1, 0))
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "the executor's edge from a stream node to a CPU node (docs/programming_model.md, \"An edge declares\")"
    )]
    fn host_wait(&self, node: NodeId) -> Status {
        gpu_status(self.events(node).1.sync())
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "the executor drains a lane at the end of a run (docs/programming_model.md, \"The orchestrator derives\")"
    )]
    fn sync(&self, lane: LaneId) -> Status {
        gpu_status(self.stream(lane).sync())
    }
    fn elapsed(&self, node: NodeId) -> Result<Option<Duration>> {
        let (start, end) = self.events(node);
        let ms = end
            .elapsed_since(start)
            .map_err(|s| gpu_status(s).expect_err("a failed status"))?;
        Duration::try_from_secs_f32(ms / 1e3)
            .map(Some)
            .map_err(|_| Error::gpu(format!("a stream scope measured {ms} ms")))
    }
}
