//! `wwr.extension.runtime:stream`: the owning stream and the borrow-safe
//! stream operations.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_event, wwr_graph, wwr_status, wwr_stream, wwr_stream_capture_mode};

use super::event::Event;
use super::first_failure;
use super::graph::Graph;
use crate::common::SourceLocation;
use crate::gpu_status;
use crate::handle::{DeviceBoundHandle, DeviceHandle, DeviceHandleStream, HandleKind};
use fb_core_error_handling::{Result, check_or_abort};

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

/// An owned stream on one device: `StreamWrapper`. Every constructor
/// aborts on failure ([`super`] has why).
///
/// It is a [`DeviceHandleStream`] on its own, so it can back a device
/// buffer with no wrapping struct, as in wwr.
pub struct Stream(DeviceBoundHandle<StreamKind>);

impl Stream {
    /// A non-blocking stream on `dev_idx`.
    #[track_caller]
    pub fn new(dev_idx: i32) -> Self {
        check_or_abort(Self::try_create(
            dev_idx,
            // SAFETY: `try_create` hands a pointer valid for one write.
            |raw| unsafe { StreamKind::create(raw) },
            Location::caller(),
        ))
    }

    /// A stream on `dev_idx` with `flags` (`wwrStreamCreateWithFlags`);
    /// pass `0` for a blocking stream.
    #[track_caller]
    pub fn with_flags(dev_idx: i32, flags: u32) -> Self {
        check_or_abort(Self::try_with_flags_at(dev_idx, flags, Location::caller()))
    }

    /// A stream on `dev_idx` with `flags` and `priority`
    /// (`wwrStreamCreateWithPriority`); a lower value is a higher priority.
    #[track_caller]
    pub fn with_priority(dev_idx: i32, flags: u32, priority: i32) -> Self {
        check_or_abort(Self::try_create(
            dev_idx,
            // SAFETY: `raw` is valid for one write.
            |raw| unsafe { wwr_sys::wwr_stream_create_with_priority(raw, flags, priority) },
            Location::caller(),
        ))
    }

    fn try_with_flags_at(dev_idx: i32, flags: u32, location: SourceLocation) -> Result<Self> {
        Self::try_create(
            dev_idx,
            // SAFETY: `raw` is valid for one write.
            |raw| unsafe { wwr_sys::wwr_stream_create_with_flags(raw, flags) },
            location,
        )
    }

    /// The fallible constructor every public one aborts over.
    fn try_create(
        dev_idx: i32,
        create: impl FnOnce(*mut wwr_stream) -> wwr_status,
        location: SourceLocation,
    ) -> Result<Self> {
        DeviceBoundHandle::try_new_at(dev_idx, create, location).map(Stream)
    }

    /// End the capture [`begin_capture`](Stream::begin_capture) started on
    /// this stream, and own the graph it recorded; a failure (no capture
    /// in progress, say) is an `Err` at the caller.
    ///
    /// An operation on a live stream, not a create, so it returns the
    /// failure rather than aborting.
    #[track_caller]
    pub fn end_capture(&self) -> Result<Graph> {
        let mut graph: wwr_graph = std::ptr::null_mut();
        // SAFETY: the borrow keeps the stream alive; writes one graph handle
        // through a valid local.
        gpu_status(unsafe { wwr_sys::wwr_stream_end_capture(self.stream(), &mut graph) })?;
        // SAFETY: the capture succeeded, so `graph` is a live graph that
        // nothing else owns.
        Ok(unsafe { Graph::adopt(graph) })
    }

    /// [`sync`] on this stream.
    pub fn sync(&self) -> wwr_status {
        // SAFETY: the borrow keeps the stream alive.
        unsafe { sync(self.stream()) }
    }

    /// [`wait_event`] on this stream, for `event`.
    pub fn wait_event(&self, event: &Event, flags: u32) -> wwr_status {
        // SAFETY: both borrows keep their handles alive.
        unsafe { wait_event(self.stream(), event.get(), flags) }
    }

    /// [`begin_capture`] on this stream; end it with
    /// [`end_capture`](Stream::end_capture).
    pub fn begin_capture(&self, mode: wwr_stream_capture_mode) -> wwr_status {
        // SAFETY: as `sync`.
        unsafe { begin_capture(self.stream(), mode) }
    }
}

impl Deref for Stream {
    type Target = DeviceBoundHandle<StreamKind>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeviceHandle for Stream {
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

impl DeviceHandleStream for Stream {
    /// The raw stream.
    fn stream(&self) -> wwr_stream {
        self.0.get()
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
    use crate::handle::tests::current_device;
    use crate::runtime::tests::{DeviceBytes, assert_gpu_error_at, ok};

    fn flags_of(stream: wwr_stream) -> u32 {
        let mut flags = u32::MAX;
        // SAFETY: writes one u32 through a valid local.
        ok(unsafe { wwr_sys::wwr_stream_get_flags(stream, &mut flags) });
        flags
    }

    /// Needs a GPU.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn the_default_stream_is_non_blocking() {
        let s = Stream::new(0);
        assert_eq!(s.dev_idx(), 0);
        assert_eq!(flags_of(s.stream()) & NON_BLOCKING, NON_BLOCKING);
        ok(s.sync());
    }

    /// Needs a GPU.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn flags_and_priority_reach_the_runtime() {
        let blocking = Stream::with_flags(0, 0);
        assert_eq!(flags_of(blocking.stream()) & NON_BLOCKING, 0);

        let (mut least, mut greatest) = (0, 0);
        // SAFETY: writes two i32s through valid locals.
        ok(unsafe { wwr_sys::wwr_device_get_stream_priority_range(&mut least, &mut greatest) });
        let s = Stream::with_priority(0, NON_BLOCKING, greatest);
        let mut priority = i32::MAX;
        // SAFETY: writes one i32 through a valid local.
        ok(unsafe { wwr_sys::wwr_stream_get_priority(s.stream(), &mut priority) });
        assert_eq!(priority, greatest);
        assert_eq!(flags_of(s.stream()) & NON_BLOCKING, NON_BLOCKING);
    }

    /// Needs a GPU. A bad index is an error at the caller, and no stream is
    /// made on whatever device was current.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn a_bad_index_is_an_error_at_the_caller() {
        let current = current_device();
        let line = line!() + 1;
        let error = Stream::try_with_flags_at(-1, 0, Location::caller())
            .err()
            .unwrap();
        assert_gpu_error_at(error, "WWR_STATUS_INVALID_DEVICE", file!(), line);
        assert_eq!(current_device(), current);
    }

    /// Needs a GPU. Capture a memset, replay it twice through the
    /// executable graph, and read the bytes back.
    ///
    /// Thread-local, not global: a global capture forbids the other
    /// threads' unsafe calls (a pool's create and destroy among them), and
    /// the other tests run beside this one. In global mode the 3080 aborted
    /// or segfaulted in `cuMemPoolDestroy` in 14 of 20 runs of the suite.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn capture_instantiate_launch() {
        let s = Stream::new(0);
        let bytes = DeviceBytes::new(16);
        ok(s.begin_capture(wwr_stream_capture_mode::WWR_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        // SAFETY: 16 bytes, queued on the capturing stream.
        ok(unsafe { wwr_sys::wwr_memset_async(bytes.0, 0x5a, 16, s.stream()) });
        let graph = s.end_capture().unwrap();

        let exec = graph.instantiate(0);
        ok(exec.upload(&s));
        ok(exec.launch(&s));
        ok(exec.launch(&s));
        assert_eq!(bytes.read(&s), [0x5a; 16]);
    }

    /// Needs a GPU. Ending a capture that never began is an error at the
    /// caller, and no graph is made.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn end_capture_without_begin_is_an_error() {
        let s = Stream::new(0);
        let line = line!() + 1;
        let error = s.end_capture().err().unwrap();
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
        assert_eq!(error.kind, fb_core_error_handling::ErrorKind::GpuRuntime);
    }
}
