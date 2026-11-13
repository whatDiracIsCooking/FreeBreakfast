//! The device side of a run.

use std::time::Duration;

use super::plan::NodeId;
#[cfg(feature = "device")]
use super::plan::{Placement, Plan};
#[cfg(feature = "device")]
use fb_core_error_handling::Error;
use fb_core_error_handling::{Result, Status};
#[cfg(feature = "device")]
use fb_core_wwr::common::ScopedDeviceIndex;
#[cfg(feature = "device")]
use fb_core_wwr::gpu_status;
#[cfg(feature = "device")]
use fb_core_wwr::runtime::{Event, Stream, event};

/// The device side of a run: what the stream workers enqueue on, and the
/// waits that make each edge leaving a stream hold
/// ([`thread_pool`](super), "When an edge is satisfied").
///
/// Every method is called from the worker the comment names, so an
/// implementation shared by reference across the run's threads is `Sync`.
pub trait Backend: Sync {
    /// What a GPU node enqueues on.
    type Stream;
    /// Held by a stream worker's thread for the whole run: its device made
    /// current. Made and dropped on that thread.
    type Binding;

    /// The number of stream workers.
    fn n_streams(&self) -> usize;
    /// On stream worker `lane`'s thread, as it starts.
    fn bind(&self, lane: usize) -> Self::Binding;
    /// Stream worker `lane`'s stream.
    fn stream(&self, lane: usize) -> &Self::Stream;
    /// On stream worker `lane`, just before GPU node `node` runs: open its
    /// stream scope.
    fn begin(&self, lane: usize, node: NodeId) -> Status;
    /// On stream worker `lane`, after GPU node `node` returned: mark the
    /// point its consumers wait for, which also closes its stream scope.
    fn record(&self, lane: usize, node: NodeId) -> Status;
    /// On stream worker `lane`, before a node that consumes `node`, which
    /// ran on another stream: make `lane` wait for it.
    fn wait(&self, lane: usize, node: NodeId) -> Status;
    /// On a CPU worker, before a node that consumes GPU node `node`: wait
    /// for it on the host.
    fn host_wait(&self, node: NodeId) -> Status;
    /// On stream worker `lane`, as it exits: wait for all its work.
    fn sync(&self, lane: usize) -> Status;
    /// After the run, every stream synced: the device time of GPU node
    /// `node`'s stream scope, which [`begin`](Backend::begin) and
    /// [`record`](Backend::record) both succeeded on in this run; `None`
    /// for a backend with no device clock.
    fn elapsed(&self, node: NodeId) -> Result<Option<Duration>>;
}

/// No device: `n` stream workers that are plain threads, whose nodes are
/// complete when they return. For device-free tests of the executor and of
/// the DAGs built on it.
#[derive(Clone, Copy, Debug)]
pub struct HostOnly {
    n_streams: usize,
}

impl HostOnly {
    /// `n_streams` stream workers.
    pub fn new(n_streams: usize) -> Self {
        HostOnly { n_streams }
    }
}

impl Backend for HostOnly {
    type Stream = ();
    type Binding = ();

    fn n_streams(&self) -> usize {
        self.n_streams
    }
    fn bind(&self, _: usize) {}
    fn stream(&self, _: usize) -> &() {
        &()
    }
    fn begin(&self, _: usize, _: NodeId) -> Status {
        Ok(())
    }
    fn record(&self, _: usize, _: NodeId) -> Status {
        Ok(())
    }
    fn wait(&self, _: usize, _: NodeId) -> Status {
        Ok(())
    }
    fn host_wait(&self, _: NodeId) -> Status {
        Ok(())
    }
    fn sync(&self, _: usize) -> Status {
        Ok(())
    }
    fn elapsed(&self, _: NodeId) -> Result<Option<Duration>> {
        Ok(None)
    }
}

/// The stream a GPU node is handed under [`Device`].
#[cfg(feature = "device")]
pub type DeviceStream = Stream;

/// Without the `device` feature: a stream that cannot exist, so the
/// planner's production defaults (`Task<Pinned, DeviceStream>`) still name a
/// type, and a task over it can be declared and planned but never run,
/// since there is no `Device` backend to run it on.
#[cfg(not(feature = "device"))]
#[derive(Debug)]
pub enum DeviceStream {}

#[cfg(feature = "device")]
/// GPU stream workers: one non-blocking stream per worker, each on its own
/// device, and two timing events per GPU node: its stream scope, recorded
/// before it runs and after it returns. The second is also what its
/// consumers wait on.
///
/// Built once per plan and reused every run: the streams and events are
/// created here, never during a run.
pub struct Device {
    devices: Vec<i32>,
    streams: Vec<DeviceStream>,
    /// Indexed by node, each GPU node's (start, end); `None` for a CPU node.
    events: Vec<Option<(Event, Event)>>,
}

#[cfg(feature = "device")]
// SAFETY: the stream and event handles are raw runtime pointers, which is
// why the wrappers are neither `Send` nor `Sync`. The runtime API is
// thread-safe for every call made on them here (record, wait, synchronize,
// elapsed time),
// and the handles are created before any run and destroyed only when the
// `Device` drops, which a run's borrow of it rules out.
unsafe impl Sync for Device {}

#[cfg(feature = "device")]
impl Device {
    /// Stream worker `s` on device `devices[s]`, with an event pair for
    /// every GPU node of `plan`.
    ///
    /// # Panics
    /// If `plan` places a node on a stream worker past `devices`.
    #[track_caller]
    pub fn new(devices: &[i32], plan: &Plan<'_, DeviceStream>) -> Self {
        assert!(
            plan.n_streams() <= devices.len(),
            "the plan uses {} stream workers but {} devices were given",
            plan.n_streams(),
            devices.len()
        );
        let streams = devices.iter().map(|&d| DeviceStream::new(d)).collect();
        let events = plan
            .nodes
            .iter()
            .map(|n| match n.placement {
                // Timing-capable: `DISABLE_TIMING` left off.
                Placement::Stream(s) => Some((
                    Event::with_flags(devices[s], event::BLOCKING_SYNC),
                    Event::with_flags(devices[s], event::BLOCKING_SYNC),
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

    fn n_streams(&self) -> usize {
        self.streams.len()
    }
    fn bind(&self, lane: usize) -> Self::Binding {
        ScopedDeviceIndex::new(self.devices[lane])
    }
    fn stream(&self, lane: usize) -> &DeviceStream {
        &self.streams[lane]
    }
    fn begin(&self, lane: usize, node: NodeId) -> Status {
        gpu_status(self.events(node).0.record(&self.streams[lane]))
    }
    fn record(&self, lane: usize, node: NodeId) -> Status {
        gpu_status(self.events(node).1.record(&self.streams[lane]))
    }
    fn wait(&self, lane: usize, node: NodeId) -> Status {
        gpu_status(self.streams[lane].wait_event(&self.events(node).1, 0))
    }
    fn host_wait(&self, node: NodeId) -> Status {
        gpu_status(self.events(node).1.sync())
    }
    fn sync(&self, lane: usize) -> Status {
        gpu_status(self.streams[lane].sync())
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
