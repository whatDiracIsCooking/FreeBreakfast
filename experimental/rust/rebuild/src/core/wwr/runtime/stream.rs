//! `wwr.extension.runtime:stream`: the owning stream and the borrow-safe
//! stream operations.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_event, wwr_graph, wwr_status, wwr_stream, wwr_stream_capture_mode};

use super::event::Event;
use super::graph::Graph;
use super::{create_with, first_failure};
use crate::core::wwr::common::{ErrorPolicy, gpu_check_at};
use crate::core::wwr::handle::{
    BaseHandle, DeviceBoundHandle, DeviceHandle, DeviceHandleStream, HandleKind,
};

/// `cudaStreamNonBlocking` / `hipStreamNonBlocking`: the stream does not
/// synchronize with the legacy default stream.
pub const NON_BLOCKING: u32 = 0x1;

/// A stream's create and destroy. The default create is non-blocking, so a
/// stream made with no flags does not serialize against the legacy default
/// stream; the destroy synchronizes first, as wwr's does.
pub struct StreamKind;

impl HandleKind for StreamKind {
    type Raw = wwr_stream;
    type Error = wwr_status;

    unsafe fn create(raw: *mut wwr_stream) -> wwr_status {
        // SAFETY: the caller hands a pointer valid for one write.
        unsafe { wwr_sys::wwr_stream_create_with_flags(raw, NON_BLOCKING) }
    }

    unsafe fn destroy(raw: wwr_stream) -> wwr_status {
        // SAFETY: `raw` is a live stream, destroyed once, after it drains.
        unsafe {
            first_failure(
                wwr_sys::wwr_stream_synchronize(raw),
                wwr_sys::wwr_stream_destroy(raw),
            )
        }
    }
}

/// An owned stream on one device: `StreamWrapper<P_create, P_destroy,
/// P_device>`.
///
/// It is a [`DeviceHandleStream`] on its own, so it can back a device
/// buffer with no wrapping struct, as in wwr.
pub struct Stream<PC, PD, PDev>(DeviceBoundHandle<StreamKind, PC, PD, PDev>)
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>;

impl<PC, PD, PDev> Stream<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// A non-blocking stream on `dev_idx`.
    #[track_caller]
    pub fn new(dev_idx: i32, policy_create: PC, policy_destroy: PD, policy_device: PDev) -> Self {
        Stream(DeviceBoundHandle::new(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
        ))
    }

    /// A stream on `dev_idx` with `flags` (`wwrStreamCreateWithFlags`);
    /// pass `0` for a blocking stream.
    #[track_caller]
    pub fn with_flags(
        dev_idx: i32,
        flags: u32,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
    ) -> Self {
        Stream(DeviceBoundHandle::new_with(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
            |policy, location| {
                create_with(policy, location, |raw| {
                    // SAFETY: `raw` is valid for one write.
                    unsafe { wwr_sys::wwr_stream_create_with_flags(raw, flags) }
                })
            },
        ))
    }

    /// A stream on `dev_idx` with `flags` and `priority`
    /// (`wwrStreamCreateWithPriority`); a lower value is a higher priority.
    #[track_caller]
    pub fn with_priority(
        dev_idx: i32,
        flags: u32,
        priority: i32,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
    ) -> Self {
        Stream(DeviceBoundHandle::new_with(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
            |policy, location| {
                create_with(policy, location, |raw| {
                    // SAFETY: `raw` is valid for one write.
                    unsafe { wwr_sys::wwr_stream_create_with_priority(raw, flags, priority) }
                })
            },
        ))
    }

    /// End the capture [`begin_capture`] started on this stream, and own
    /// the graph it recorded.
    ///
    /// A failure goes to this stream's create policy, at the caller, as
    /// wwr's does; the graph then holds nothing. The graph's own policies
    /// are new defaults of this stream's types, as wwr default-constructs
    /// them. It takes `&mut self` because the create policy may record.
    #[track_caller]
    pub fn end_capture(&mut self) -> Graph<PC, PD>
    where
        PC: Default,
        PD: Default,
    {
        let location = Location::caller();
        let stream = self.stream();
        let mut graph: wwr_graph = std::ptr::null_mut();
        // SAFETY: writes one graph handle through a valid local.
        let ok = gpu_check_at(
            unsafe { wwr_sys::wwr_stream_end_capture(stream, &mut graph) },
            self.0.create_policy_mut(),
            location,
        );
        let captured = ok.then_some(graph);
        Graph(BaseHandle::new_with(
            PC::default(),
            PD::default(),
            |_, _| captured,
        ))
    }
}

impl<PC, PD, PDev> Deref for Stream<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    type Target = DeviceBoundHandle<StreamKind, PC, PD, PDev>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<PC, PD, PDev> DeviceHandle for Stream<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

impl<PC, PD, PDev> DeviceHandleStream for Stream<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The raw stream, or null (the legacy default stream) when the create
    /// failed under a policy that returns: what wwr's `stream()` hands back
    /// from its null sentinel.
    fn stream(&self) -> wwr_stream {
        self.0.get().unwrap_or(std::ptr::null_mut())
    }
}

impl<PC, PD, PDev> Stream<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// [`sync`] on this stream.
    pub fn sync(&self) -> wwr_status {
        // SAFETY: the borrow keeps the stream alive (or it is null, the
        // legacy default stream).
        unsafe { sync(self.stream()) }
    }

    /// [`wait_event`] on this stream, for `event`; `WWR_STATUS_INVALID_VALUE`
    /// if `event` owns nothing.
    pub fn wait_event<EC, ED, EDev>(&self, event: &Event<EC, ED, EDev>, flags: u32) -> wwr_status
    where
        EC: ErrorPolicy<Error = wwr_status>,
        ED: ErrorPolicy<Error = wwr_status>,
        EDev: ErrorPolicy<Error = wwr_status>,
    {
        match event.get() {
            // SAFETY: both borrows keep their handles alive.
            Some(raw) => unsafe { wait_event(self.stream(), raw, flags) },
            None => wwr_status::WWR_STATUS_INVALID_VALUE,
        }
    }

    /// [`begin_capture`] on this stream; end it with
    /// [`end_capture`](Stream::end_capture).
    pub fn begin_capture(&self, mode: wwr_stream_capture_mode) -> wwr_status {
        // SAFETY: as `sync`.
        unsafe { begin_capture(self.stream(), mode) }
    }
}

// The borrow-safe operations on a raw stream, as wwr's free functions. They
// are `unsafe` where wwr's are not: a destroyed handle is a use-after-free
// inside the runtime, and only an owner's borrow rules that out. An owner
// calls them through its methods above.

/// Make `stream` wait for `event` before running later work.
///
/// # Safety
/// `stream` is a live stream or null (the legacy default stream), and
/// `event` a live event.
pub unsafe fn wait_event(stream: wwr_stream, event: wwr_event, flags: u32) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_stream_wait_event(stream, event, flags) }
}

/// Start recording the work queued on `stream` into a graph.
///
/// # Safety
/// `stream` is a live stream.
pub unsafe fn begin_capture(stream: wwr_stream, mode: wwr_stream_capture_mode) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_stream_begin_capture(stream, mode) }
}

/// Wait for the work queued on `stream` to finish.
///
/// # Safety
/// `stream` is a live stream or null (the legacy default stream).
pub unsafe fn sync(stream: wwr_stream) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_stream_synchronize(stream) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wwr::common::tests::Recording;
    use crate::core::wwr::handle::tests::current_device;
    use crate::core::wwr::runtime::tests::{Abort, DeviceBytes, ok};

    type Recorded = Stream<Recording, Recording, Recording>;

    fn flags_of(stream: wwr_stream) -> u32 {
        let mut flags = u32::MAX;
        // SAFETY: writes one u32 through a valid local.
        ok(unsafe { wwr_sys::wwr_stream_get_flags(stream, &mut flags) });
        flags
    }

    /// Needs a GPU.
    #[test]
    fn the_default_stream_is_non_blocking() {
        let s = Stream::new(0, Abort::new(), Abort::new(), Abort::new());
        assert_eq!(s.dev_idx(), 0);
        assert_eq!(flags_of(s.stream()) & NON_BLOCKING, NON_BLOCKING);
        ok(s.sync());
    }

    /// Needs a GPU.
    #[test]
    fn flags_and_priority_reach_the_runtime() {
        let blocking = Stream::with_flags(0, 0, Abort::new(), Abort::new(), Abort::new());
        assert_eq!(flags_of(blocking.stream()) & NON_BLOCKING, 0);

        let (mut least, mut greatest) = (0, 0);
        // SAFETY: writes two i32s through valid locals.
        ok(unsafe { wwr_sys::wwr_device_get_stream_priority_range(&mut least, &mut greatest) });
        let s = Stream::with_priority(
            0,
            NON_BLOCKING,
            greatest,
            Abort::new(),
            Abort::new(),
            Abort::new(),
        );
        let mut priority = i32::MAX;
        // SAFETY: writes one i32 through a valid local.
        ok(unsafe { wwr_sys::wwr_stream_get_priority(s.stream(), &mut priority) });
        assert_eq!(priority, greatest);
        assert_eq!(flags_of(s.stream()) & NON_BLOCKING, NON_BLOCKING);
    }

    /// Needs a GPU. A bad index goes to the device policy; the stream lands
    /// on the current device.
    #[test]
    fn a_bad_index_is_reported_at_the_caller() {
        let current = current_device();
        let line = line!() + 1;
        let s = Recorded::with_flags(
            -1,
            0,
            Recording::default(),
            Recording::default(),
            Recording::default(),
        );
        assert_eq!(
            s.device_policy().0,
            [(wwr_status::WWR_STATUS_INVALID_DEVICE, line)]
        );
        assert_eq!(s.dev_idx(), current);
    }

    /// Needs a GPU. Capture a memset, replay it twice through the
    /// executable graph, and read the bytes back.
    ///
    /// Thread-local, not global: a global capture forbids the other
    /// threads' unsafe calls (a pool's create and destroy among them), and
    /// the other tests run beside this one. In global mode the 3080 aborted
    /// or segfaulted in `cuMemPoolDestroy` in 14 of 20 runs of the suite.
    #[test]
    fn capture_instantiate_launch() {
        let mut s = Stream::new(0, Abort::new(), Abort::new(), Abort::new());
        let bytes = DeviceBytes::new(16);
        ok(s.begin_capture(wwr_stream_capture_mode::WWR_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        // SAFETY: 16 bytes, queued on the capturing stream.
        ok(unsafe { wwr_sys::wwr_memset_async(bytes.0, 0x5a, 16, s.stream()) });
        let graph = s.end_capture();
        assert!(graph.get().is_some());

        let exec = graph.instantiate(0);
        ok(exec.upload(&s));
        ok(exec.launch(&s));
        ok(exec.launch(&s));
        assert_eq!(bytes.read(s.stream()), [0x5a; 16]);
    }

    /// Needs a GPU. Ending a capture that never began fails through the
    /// stream's create policy, and the graph owns nothing.
    #[test]
    fn end_capture_without_begin_is_reported() {
        let mut s = Recorded::new(
            0,
            Recording::default(),
            Recording::default(),
            Recording::default(),
        );
        let line = line!() + 1;
        let graph = s.end_capture();
        assert_eq!(graph.get(), None);
        let reported = &s.create_policy().0;
        assert_eq!(reported.len(), 1);
        assert_ne!(reported[0].0, wwr_status::WWR_STATUS_SUCCESS);
        assert_eq!(reported[0].1, line);
    }
}
