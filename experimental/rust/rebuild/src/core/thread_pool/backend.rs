//! The device side of a run.

use wwr_sys::wwr_status;

use super::plan::{NodeId, Placement, Plan};
use crate::core::error_handling::Status;
use crate::core::wwr::common::ScopedDeviceIndex;
use crate::core::wwr::runtime::{Event, Stream, event};
use crate::core::wwr::{CreatePolicy, DestroyPolicy, gpu_status};

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
    /// On stream worker `lane`, after GPU node `node` returned: mark the
    /// point its consumers wait for.
    fn record(&self, lane: usize, node: NodeId) -> Status;
    /// On stream worker `lane`, before a node that consumes `node`, which
    /// ran on another stream: make `lane` wait for it.
    fn wait(&self, lane: usize, node: NodeId) -> Status;
    /// On a CPU worker, before a node that consumes GPU node `node`: wait
    /// for it on the host.
    fn host_wait(&self, node: NodeId) -> Status;
    /// On stream worker `lane`, as it exits: wait for all its work.
    fn sync(&self, lane: usize) -> Status;
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
}

type Create = CreatePolicy<wwr_status>;
type Destroy = DestroyPolicy<wwr_status>;

/// The stream a GPU node is handed under [`Device`].
pub type DeviceStream = Stream<Create, Destroy, Create>;
type DeviceEvent = Event<Create, Destroy, Create>;

/// GPU stream workers: one non-blocking stream per worker, each on its own
/// device, and one event per GPU node, recorded after it returns.
///
/// Built once per plan and reused every run: the streams and events are
/// created here, never during a run.
pub struct Device {
    devices: Vec<i32>,
    streams: Vec<DeviceStream>,
    /// Indexed by node; `None` for a CPU node.
    events: Vec<Option<DeviceEvent>>,
}

// SAFETY: the stream and event handles are raw runtime pointers, which is
// why the wrappers are neither `Send` nor `Sync`. The runtime API is
// thread-safe for every call made on them here (record, wait, synchronize),
// the handles are created before any run and destroyed only when the
// `Device` drops, which a run's borrow of it rules out, and the zero-sized
// policies they carry hold no state.
unsafe impl Sync for Device {}

impl Device {
    /// Stream worker `s` on device `devices[s]`, with an event for every GPU
    /// node of `plan`.
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
        let streams = devices
            .iter()
            .map(|&d| DeviceStream::new(d, Create::new(), Destroy::new(), Create::new()))
            .collect();
        let events = plan
            .nodes
            .iter()
            .map(|n| match n.placement {
                Placement::Stream(s) => Some(DeviceEvent::with_flags(
                    devices[s],
                    event::DISABLE_TIMING | event::BLOCKING_SYNC,
                    Create::new(),
                    Destroy::new(),
                    Create::new(),
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

    fn event(&self, node: NodeId) -> &DeviceEvent {
        self.events[node.index()]
            .as_ref()
            .expect("an event is recorded only for a GPU node")
    }
}

impl Backend for Device {
    type Stream = DeviceStream;
    type Binding = ScopedDeviceIndex<Create>;

    fn n_streams(&self) -> usize {
        self.streams.len()
    }
    fn bind(&self, lane: usize) -> Self::Binding {
        ScopedDeviceIndex::new(self.devices[lane], Create::new())
    }
    fn stream(&self, lane: usize) -> &DeviceStream {
        &self.streams[lane]
    }
    fn record(&self, lane: usize, node: NodeId) -> Status {
        gpu_status(self.event(node).record(&self.streams[lane]))
    }
    fn wait(&self, lane: usize, node: NodeId) -> Status {
        gpu_status(self.streams[lane].wait_event(self.event(node), 0))
    }
    fn host_wait(&self, node: NodeId) -> Status {
        gpu_status(self.event(node).sync())
    }
    fn sync(&self, lane: usize) -> Status {
        gpu_status(self.streams[lane].sync())
    }
}
