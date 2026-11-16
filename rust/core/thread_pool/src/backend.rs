//! The device side of a run.

use std::time::Duration;

use super::task_graph::{LaneId, NodeId};
#[cfg(feature = "device")]
use super::task_graph::{Placement, TaskGraph};
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
    /// On `lane`'s worker, just before GPU node `node` runs on it in trip
    /// `trip`: open its stream scope. A node's trip is its iteration of
    /// the counted loop it is in, each with events of its own, and 0
    /// outside one: a decision loop's iterations reuse one pair, read by
    /// the decision before the next ([`thread_pool`](super), "Timing").
    fn begin(&self, lane: LaneId, node: NodeId, trip: usize) -> Status;
    /// On `lane`'s worker, after GPU node `node` returned in trip `trip`:
    /// mark the point its consumers wait for, which also closes its stream
    /// scope.
    fn record(&self, lane: LaneId, node: NodeId, trip: usize) -> Status;
    /// On `lane`'s worker, before a node on `lane` that consumes `node`'s
    /// trip `trip`, which ran on another lane: make `lane` wait for it on
    /// the device, with no host wait.
    fn wait(&self, lane: LaneId, node: NodeId, trip: usize) -> Status;
    /// On a CPU worker, before a node that consumes GPU node `node`'s trip
    /// `trip`: wait for it on the host.
    fn host_wait(&self, node: NodeId, trip: usize) -> Status;
    /// On `lane`'s worker, as it exits: wait for all the lane's work, and
    /// for the worker's fork streams' (`DeviceStream::fork_join`).
    fn sync(&self, lane: LaneId) -> Status;
    /// After the run, every stream synced: the device time of GPU node
    /// `node`'s stream scope in trip `trip`, which
    /// [`begin`](Backend::begin) and [`record`](Backend::record) both
    /// succeeded on in this run; `None` for a backend with no device
    /// clock.
    fn elapsed(&self, node: NodeId, trip: usize) -> Result<Option<Duration>>;
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
    fn begin(&self, _: LaneId, _: NodeId, _: usize) -> Status {
        Ok(())
    }
    fn record(&self, _: LaneId, _: NodeId, _: usize) -> Status {
        Ok(())
    }
    fn wait(&self, _: LaneId, _: NodeId, _: usize) -> Status {
        Ok(())
    }
    fn host_wait(&self, _: NodeId, _: usize) -> Status {
        Ok(())
    }
    fn sync(&self, _: LaneId) -> Status {
        Ok(())
    }
    fn elapsed(&self, _: NodeId, _: usize) -> Result<Option<Duration>> {
        Ok(None)
    }
}

/// What a GPU node is handed under [`Device`]: its lane's stream, which it
/// derefs to, and the lane's library handles, each bound to that stream, if
/// the lane has them ([`Device::new`]).
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
///
/// A node may also fork its own work across its worker's fork streams and
/// join it back before it returns ([`DeviceStream::fork_join`]).
#[cfg(feature = "device")]
pub struct DeviceStream {
    // Fields drop in declaration order: the handles before the stream they
    // are bound to (each also holds the stream until it is destroyed).
    // `None` on a lane made without them.
    blas: Option<BlasHandle>,
    solver: Option<SolverHandle>,
    stream: Rc<Stream>,
    /// The worker's fork streams, shared by its lanes; `None` for a worker
    /// none of whose tasks forks onto two or more.
    forks: Option<Rc<Forks>>,
}

/// A GPU worker's fork streams ([`DeviceStream::fork_join`]): outside its
/// lanes, with no library handles, made with the backend and shared by the
/// worker's lanes, which its one thread drives one node at a time.
#[cfg(feature = "device")]
struct Forks {
    /// Recorded on the forking stream; every fork waits on it.
    fork: Event,
    /// Each fork's stream, and the event its join records on it, which the
    /// forking stream waits on.
    streams: Vec<(Stream, Event)>,
}

#[cfg(feature = "device")]
impl Forks {
    /// `n` non-blocking streams on `dev_idx` and their events, none of
    /// them timed. Aborts if any cannot be made.
    #[track_caller]
    fn new(dev_idx: i32, n: usize) -> Self {
        let untimed = || Event::with_flags(dev_idx, event::DISABLE_TIMING);
        Forks {
            fork: untimed(),
            streams: (0..n).map(|_| (Stream::new(dev_idx), untimed())).collect(),
        }
    }
}

#[cfg(feature = "device")]
impl DeviceStream {
    /// A non-blocking stream on `dev_idx`, and a BLAS and a dense-solver
    /// handle bound to it, with no fork streams. Aborts if any cannot be
    /// made.
    #[track_caller]
    pub fn new(dev_idx: i32) -> Self {
        Self::with_libraries(dev_idx, true, None)
    }

    /// A non-blocking stream on `dev_idx`, its two handles if `libraries`,
    /// and its worker's `forks`. Aborts if any cannot be made.
    #[track_caller]
    fn with_libraries(dev_idx: i32, libraries: bool, forks: Option<Rc<Forks>>) -> Self {
        let stream = Rc::new(Stream::new(dev_idx));
        DeviceStream {
            blas: libraries.then(|| BlasHandle::new(Rc::clone(&stream))),
            solver: libraries.then(|| SolverHandle::new(Rc::clone(&stream))),
            stream,
            forks,
        }
    }

    /// Run `f(i, fork)` for every `i` in `0..n`, each enqueueing on `fork`,
    /// a stream forked from this one, and join every fork back into this
    /// stream before returning (`docs/programming_model.md`, "Forks inside
    /// a task").
    ///
    /// Every fork waits on the device for all the work already on this
    /// stream, the task's input waits and whatever it enqueued before the
    /// call, and this stream then waits on the device for every fork, so
    /// what is enqueued on it afterwards, the task's own later launches,
    /// the event the executor records when the task returns and every
    /// edge and handoff ordered by that event, follows every fork's work.
    /// No wait is on the host. The items are not ordered among
    /// themselves, so they must not conflict: the one-electron class
    /// kernels add atomically into one matrix, which is the case this is
    /// for.
    ///
    /// The forks are the worker's fork streams, as many as its most
    /// forking task declared at planning (the planner's `GpuTask::forks`),
    /// item `i` on fork `i mod k` where `k` is the smaller of `n` and that
    /// count; with `k` below two, every item runs on this stream itself,
    /// in order, and nothing is forked. The forks are not other lanes'
    /// streams, so a fork orders the task behind nothing the planner did
    /// not order it behind; two nodes of one worker that fork share them,
    /// and their forked items keep the order the worker enqueued them in.
    ///
    /// # Errors
    /// The first failure of the fork, of an `f(i, ..)` or of a join. After
    /// a failure no further item runs, but every fork already made is
    /// still joined back, so this stream never runs ahead of an item in
    /// flight.
    ///
    /// ```no_run
    /// use fb_core_error_handling::Status;
    /// use fb_core_thread_pool::DeviceStream;
    /// use fb_core_wwr::runtime::Stream;
    ///
    /// fn launch(class: usize, on: &Stream) -> Status {
    ///     // A class's kernel, enqueued on `on`.
    ///     Ok(())
    /// }
    ///
    /// let stream = DeviceStream::new(0);
    /// stream.fork_join(4, |c, fork| launch(c, fork))?;
    /// // Enqueued after every class's work.
    /// # Ok::<(), fb_core_error_handling::Error>(())
    /// ```
    ///
    /// A fork is a borrow for the call, so none outlives its join:
    ///
    /// ```compile_fail,E0521
    /// use fb_core_thread_pool::DeviceStream;
    /// use fb_core_wwr::runtime::Stream;
    ///
    /// let stream = DeviceStream::new(0);
    /// let mut kept: Option<&Stream> = None;
    /// stream.fork_join(2, |_, fork| {
    ///     kept = Some(fork);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), fb_core_error_handling::Error>(())
    /// ```
    pub fn fork_join<F>(&self, n: usize, mut f: F) -> Status
    where
        F: FnMut(usize, &Stream) -> Status,
    {
        let Some(pool) = self.forks.as_deref().filter(|_| n >= 2) else {
            return (0..n).try_for_each(|i| f(i, &self.stream));
        };
        let forks = &pool.streams[..pool.streams.len().min(n)];
        gpu_status(pool.fork.record(&self.stream))?;
        let mut made = 0;
        let mut result = Ok(());
        for (fork, _) in forks {
            result = gpu_status(fork.wait_event(&pool.fork, 0));
            if result.is_err() {
                break;
            }
            made += 1;
        }
        if result.is_ok() {
            result = (0..n).try_for_each(|i| f(i, &forks[i % forks.len()].0));
        }
        for (fork, join) in &forks[..made] {
            let joined = gpu_status(join.record(fork))
                .and_then(|()| gpu_status(self.stream.wait_event(join, 0)));
            result = result.and(joined);
        }
        result
    }

    /// The lane's BLAS handle, bound to its stream: pass its `get()` to
    /// a `wwr_blas_*` call, and the call enqueues on the stream.
    ///
    /// # Panics
    /// If the lane was made without handles: a task on it said it calls no
    /// library (the planner's `GpuTask::uses_libraries`) and calls one.
    pub fn blas(&self) -> &BlasHandle {
        self.blas
            .as_ref()
            .expect("a lane made without library handles: its tasks said they call none")
    }

    /// The lane's dense-solver handle, bound to its stream: pass its
    /// `get()` to a `wwr_solver_dn_*` call, and the call enqueues on the
    /// stream.
    ///
    /// # Panics
    /// As [`DeviceStream::blas`].
    pub fn solver(&self) -> &SolverHandle {
        self.solver
            .as_ref()
            .expect("a lane made without library handles: its tasks said they call none")
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
/// lane and, on the lanes that call a library, the lane's library handles
/// bound to it ([`DeviceStream`]), its fork streams if a task forks
/// ([`DeviceStream::fork_join`]), and
/// two timing events per GPU node and trip: its stream scope, recorded
/// before it runs and after it returns. The second is also what its
/// consumers wait on. A node outside every counted loop has one trip; one
/// in a counted loop a pair per iteration, so each iteration's scope is
/// read after the run and no wait ever finds its event recorded again by
/// a later iteration of the same run.
///
/// Built once per plan and reused every run, one run at a time
/// ([`run`](super::run) borrows it exclusively): the streams, fork
/// streams, handles and events are created here, never during a run.
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
    /// Indexed by node and then trip, each GPU node's (start, end); empty
    /// for a CPU node.
    events: Vec<Vec<(Event, Event)>>,
}

#[cfg(feature = "device")]
impl Device {
    /// GPU worker `w` on device `devices[w]`, with as many lanes as `plan`
    /// places nodes on ([`TaskGraph::lanes`]), at least one, and an event pair
    /// for every GPU node of `plan` and trip of the counted loop it is in.
    /// Lane `l` of worker `w` has library handles if `libraries[w][l]`, and
    /// none if it is false or missing, so a lane whose nodes call no
    /// library costs no handle and no library workspace. Worker `w` has
    /// `forks[w]` fork streams ([`DeviceStream::fork_join`]), outside its
    /// lanes and with no library handles, and none if that is below two or
    /// missing, since one fork would only serialize what this stream does.
    #[debug_requires(
        plan.lanes().len() <= devices.len(),
        "a device for every GPU worker of the plan"
    )]
    #[track_caller]
    pub fn new(
        devices: &[i32],
        plan: &TaskGraph<'_, DeviceStream>,
        libraries: &[Vec<bool>],
        forks: &[usize],
    ) -> Self {
        let lanes = plan.lanes();
        let streams = devices
            .iter()
            .enumerate()
            .map(|(w, &d)| {
                let n = lanes.get(w).copied().unwrap_or(0).max(1);
                let has = |l: usize| libraries.get(w).and_then(|v| v.get(l)).copied();
                let pool = forks
                    .get(w)
                    .copied()
                    .filter(|&k| k >= 2)
                    .map(|k| Rc::new(Forks::new(d, k)));
                (0..n)
                    .map(|l| DeviceStream::with_libraries(d, has(l).unwrap_or(false), pool.clone()))
                    .collect()
            })
            .collect();
        let events = plan
            .nodes
            .iter()
            .map(|n| match n.placement {
                // Timing-capable: `DISABLE_TIMING` left off.
                Placement::Stream(at) => {
                    let pair = || {
                        (
                            Event::with_flags(devices[at.worker], event::BLOCKING_SYNC),
                            Event::with_flags(devices[at.worker], event::BLOCKING_SYNC),
                        )
                    };
                    (0..plan.trips(n)).map(|_| pair()).collect()
                }
                Placement::Cpu => Vec::new(),
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
    /// If the backend does not have `lane`, or `lane` has no library
    /// handles ([`Device::new`]).
    pub fn bind_blas_workspace<W>(&mut self, lane: LaneId, workspace: W) -> Status
    where
        W: DeviceSliceMut<u8> + Send + Sync + 'static,
    {
        self.streams[lane.worker][lane.lane]
            .blas
            .as_mut()
            .expect("a lane made without library handles has no workspace to bind")
            .bind_workspace(workspace)
    }

    fn events(&self, node: NodeId, trip: usize) -> &(Event, Event) {
        self.events[node.index()]
            .get(trip)
            .expect("an event is recorded only for a GPU node, in a trip it has")
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
    fn begin(&self, lane: LaneId, node: NodeId, trip: usize) -> Status {
        gpu_status(self.events(node, trip).0.record(self.stream(lane)))
    }
    fn record(&self, lane: LaneId, node: NodeId, trip: usize) -> Status {
        gpu_status(self.events(node, trip).1.record(self.stream(lane)))
    }
    fn wait(&self, lane: LaneId, node: NodeId, trip: usize) -> Status {
        gpu_status(self.stream(lane).wait_event(&self.events(node, trip).1, 0))
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "the executor's edge from a stream node to a CPU node (docs/programming_model.md, \"An edge declares\")"
    )]
    fn host_wait(&self, node: NodeId, trip: usize) -> Status {
        gpu_status(self.events(node, trip).1.sync())
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "the executor drains a lane at the end of a run (docs/programming_model.md, \"The orchestrator derives\")"
    )]
    fn sync(&self, lane: LaneId) -> Status {
        let stream = self.stream(lane);
        gpu_status(stream.sync())?;
        // The worker's forks too: each fork_join joins its forks back into
        // the forking stream, so they are idle here unless a join failed.
        stream
            .forks
            .iter()
            .flat_map(|f| &f.streams)
            .try_for_each(|(fork, _)| gpu_status(fork.sync()))
    }
    fn elapsed(&self, node: NodeId, trip: usize) -> Result<Option<Duration>> {
        let (start, end) = self.events(node, trip);
        let ms = end
            .elapsed_since(start)
            .map_err(|s| gpu_status(s).expect_err("a failed status"))?;
        Duration::try_from_secs_f32(ms / 1e3)
            .map(Some)
            .map_err(|_| Error::gpu(format!("a stream scope measured {ms} ms")))
    }
}
