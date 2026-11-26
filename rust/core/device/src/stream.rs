//! A stream on one device, its copies and fills, and an event.

use std::ffi::c_void;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use contracts::requires;
use fb_core_error_handling::{Status, check_or_abort};
use fb_core_log::le;
use wwr_sys::{wwr_event, wwr_memcpy_kind, wwr_stream};

use crate::slice::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut};
use crate::{ScopedDevice, log_destroy, runtime_status};

/// `cudaStreamNonBlocking` / `hipStreamNonBlocking`: the stream does not
/// synchronize with the legacy default stream.
const NON_BLOCKING: u32 = 0x1;

/// The stream syncs of the process so far, and the nanoseconds blocked in
/// them.
static SYNCS: AtomicU64 = AtomicU64::new(0);
static SYNC_NANOS: AtomicU64 = AtomicU64::new(0);

/// How many times [`Stream::sync`] has been called so far, by any thread,
/// and the time its callers spent blocked there. A measurement, read as a
/// difference over a span in which one step runs.
pub fn sync_meter() -> (u64, Duration) {
    (
        SYNCS.load(Ordering::Relaxed),
        Duration::from_nanos(SYNC_NANOS.load(Ordering::Relaxed)),
    )
}

/// A non-blocking stream on one device, destroyed on drop.
///
/// `Send` and `Sync`: a step runs on whichever worker picks it up, and its
/// streams move with it.
pub struct Stream {
    raw: wwr_stream,
    physical: i32,
}

// SAFETY: the raw stream is a runtime object, not tied to the thread that
// made it, and the runtime API is thread-safe: every call names the stream,
// and each method here makes its device current first. Moving the owner
// moves the pointer alone, destroyed once, by its drop.
unsafe impl Send for Stream {}
// SAFETY: as above. Every `&self` method is one enqueue, a wait or a sync
// on the live stream, which the runtime allows from several threads at
// once; the order of work two threads enqueue meanwhile is the callers' to
// keep, not memory safety. The drop takes the stream by value.
unsafe impl Sync for Stream {}

/// An event on one device, destroyed on drop: a point on a stream that
/// another stream, or the host, waits for. `Send` and `Sync`, as
/// [`Stream`] is.
pub struct Event {
    raw: wwr_event,
    physical: i32,
}

// SAFETY: as `Stream`'s: a runtime object any thread may name, each method
// making its device current first.
unsafe impl Send for Event {}
// SAFETY: as `Stream`'s: a record, a wait on it or a sync is one runtime
// call, which the runtime allows from several threads at once.
unsafe impl Sync for Event {}

// The four above, checked at every build.
const _: () = {
    const fn shared_across_threads<T: Send + Sync>() {}
    shared_across_threads::<Stream>();
    shared_across_threads::<Event>();
};

impl Stream {
    /// A non-blocking stream on device `physical`, aborting at the caller's
    /// line if the runtime refuses.
    #[track_caller]
    pub fn new(physical: i32) -> Stream {
        let _current = ScopedDevice::new(physical);
        let mut raw: wwr_stream = std::ptr::null_mut();
        // SAFETY: writes one handle through a valid local, with the device
        // current.
        let status = unsafe { wwr_sys::wwr_stream_create_with_flags(&mut raw, NON_BLOCKING) };
        check_or_abort(runtime_status(status));
        Stream { raw, physical }
    }

    /// The handle, for a launch wrapper's `extern "C"` call.
    pub fn raw(&self) -> wwr_stream {
        self.raw
    }

    /// The device the stream is on.
    pub fn physical(&self) -> i32 {
        self.physical
    }

    /// Block until everything enqueued has run, counted and timed in
    /// [`sync_meter`].
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced here.
    #[track_caller]
    pub fn sync(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        let start = Instant::now();
        #[allow(
            clippy::disallowed_methods,
            reason = "the step wrapper's sync after `run`, the v0.0.4 layer's edge (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        // SAFETY: a live stream.
        let status = unsafe { wwr_sys::wwr_stream_synchronize(self.raw) };
        SYNCS.fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        SYNC_NANOS.fetch_add(nanos, Ordering::Relaxed);
        runtime_status(status)
    }

    /// Make everything enqueued after this wait for `event`'s recorded
    /// point, which may be another device's.
    ///
    /// # Errors
    /// The runtime's failure.
    #[track_caller]
    pub fn wait(&self, event: &Event) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a live stream and a live event.
        runtime_status(unsafe { wwr_sys::wwr_stream_wait_event(self.raw, event.raw, 0) })
    }

    /// Enqueue the zeroing of `dst`.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[track_caller]
    pub fn zero<T>(&self, dst: &mut impl DeviceSliceMut<T>) -> Status {
        self.fill(dst, 0)
    }

    /// Enqueue the setting of every byte of `dst` to `byte` (`0xFF` is a NaN
    /// in every double).
    ///
    /// # Errors
    /// The enqueue's failure.
    #[track_caller]
    pub fn fill<T>(&self, dst: &mut impl DeviceSliceMut<T>, byte: u8) -> Status {
        if dst.is_empty() {
            return Ok(());
        }
        let _current = ScopedDevice::new(self.physical);
        // No overflow: the slice promises `len()` elements of memory.
        let bytes = dst.len() * size_of::<T>();
        // SAFETY: `dst` is `bytes` of device memory with no other user
        // until the work enqueued during its borrow finishes
        // (`DeviceSliceMut`'s promise); the stream is live.
        runtime_status(unsafe {
            wwr_sys::wwr_memset_async(dst.as_mut_ptr().cast(), i32::from(byte), bytes, self.raw)
        })
    }

    /// Enqueue the copy of `src` into the head of device `dst`, its first
    /// `src.len()` elements; the rest of `dst` is left as it is.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[requires(le!(src.len(), dst.len()), "the device end holds the host end")]
    #[track_caller]
    pub fn upload<T: Copy>(
        &self,
        dst: &mut impl DeviceSliceMut<T>,
        src: &impl HostSlice<T>,
    ) -> Status {
        // No overflow: the host slice promises `len()` elements of memory.
        let bytes = src.len() * size_of::<T>();
        // SAFETY: `src` is `bytes` of host memory nothing writes, and `dst`
        // holds at least as many of device memory with no other user, both
        // until the work enqueued here finishes (the traits' promises).
        unsafe {
            self.copy(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            )
        }
    }

    /// Enqueue the copy of the head of device `src`, its first `dst.len()`
    /// elements, into `dst`, which is read only after the copy has run.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[requires(le!(dst.len(), src.len()), "the device end holds the host end")]
    #[track_caller]
    pub fn download<T: Copy>(
        &self,
        dst: &mut impl HostSliceMut<T>,
        src: &impl DeviceSlice<T>,
    ) -> Status {
        // No overflow: the host slice promises `len()` elements of memory.
        let bytes = dst.len() * size_of::<T>();
        // SAFETY: `dst` is `bytes` of host memory with no other user, and
        // `src` at least as many of device memory nothing writes, both
        // until the work enqueued here finishes (the traits' promises).
        unsafe {
            self.copy(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
            )
        }
    }

    /// Enqueue the copy of device `src` into the head of device `dst`, its
    /// first `src.len()` elements. Two parts of one slice are apart, so a
    /// copy between them is too ([`split_at_mut`](crate::split_at_mut)).
    ///
    /// # Errors
    /// The enqueue's failure.
    #[requires(le!(src.len(), dst.len()), "the destination holds the source")]
    #[track_caller]
    pub fn copy_device<T: Copy>(
        &self,
        dst: &mut impl DeviceSliceMut<T>,
        src: &impl DeviceSlice<T>,
    ) -> Status {
        // No overflow: the slice promises `len()` elements of memory.
        let bytes = src.len() * size_of::<T>();
        // SAFETY: `src` is `bytes` of device memory nothing writes, and
        // `dst` at least as many with no other user, so apart from `src`,
        // both until the work enqueued here finishes (the traits' promises).
        unsafe {
            self.copy(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
            )
        }
    }

    /// The one `wwr_memcpy_async` of the three copies above; nothing is
    /// enqueued for no bytes.
    ///
    /// # Safety
    /// `src` and `dst` each hold `bytes` of memory of the side `kind` names,
    /// valid until the work enqueued here finishes; nothing writes `src`
    /// meanwhile, and nothing but the copy uses `dst`.
    #[track_caller]
    unsafe fn copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: wwr_memcpy_kind,
    ) -> Status {
        if bytes == 0 {
            return Ok(());
        }
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: the caller's promise for both ends; the stream is live.
        runtime_status(unsafe { wwr_sys::wwr_memcpy_async(dst, src, bytes, kind, self.raw) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let location = Location::caller();
        let current = ScopedDevice::try_new(self.physical);
        log_destroy("stream destroy", &current, location);
        // SAFETY: created by `wwr_stream_create_with_flags` in `new`, and
        // destroyed once, here; work still enqueued completes first.
        let status = unsafe { wwr_sys::wwr_stream_destroy(self.raw) };
        log_destroy("stream destroy", &runtime_status(status), location);
    }
}

impl Event {
    /// An event on device `physical`, aborting at the caller's line if the
    /// runtime refuses.
    #[track_caller]
    pub fn new(physical: i32) -> Event {
        let _current = ScopedDevice::new(physical);
        let mut raw: wwr_event = std::ptr::null_mut();
        // SAFETY: writes one handle through a valid local, with the device
        // current.
        let status = unsafe { wwr_sys::wwr_event_create(&mut raw) };
        check_or_abort(runtime_status(status));
        Event { raw, physical }
    }

    /// Mark the end of what `stream`, a stream of the event's device, holds
    /// now.
    ///
    /// # Errors
    /// The runtime's failure, a stream of another device among them.
    #[track_caller]
    pub fn record(&self, stream: &Stream) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a live event and a live stream.
        runtime_status(unsafe { wwr_sys::wwr_event_record(self.raw, stream.raw) })
    }

    /// Block the host until the recorded point has run.
    ///
    /// # Errors
    /// A fault of the work before it, surfaced here.
    #[track_caller]
    pub fn sync(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        #[allow(
            clippy::disallowed_methods,
            reason = "the host's wait on a ring buffer's last event before it rewrites its pinned bytes (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        // SAFETY: a live event.
        let status = unsafe { wwr_sys::wwr_event_synchronize(self.raw) };
        runtime_status(status)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let location = Location::caller();
        let current = ScopedDevice::try_new(self.physical);
        log_destroy("event destroy", &current, location);
        // SAFETY: created by `wwr_event_create` in `new`, and destroyed
        // once, here.
        let status = unsafe { wwr_sys::wwr_event_destroy(self.raw) };
        log_destroy("event destroy", &runtime_status(status), location);
    }
}
