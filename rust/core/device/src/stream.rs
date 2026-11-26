//! A stream on one device, its copies and fills, and an event.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use contracts::requires;
use fb_core_error_handling::{Result, Status, check_or_abort};
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
    /// The syncs begun on the stream, each taking the count before it as
    /// its ticket; a copy's [`Pending`] takes the count after its enqueue.
    pub(crate) tickets: AtomicU64,
    /// One past the largest ticket of a sync that succeeded: a copy whose
    /// ticket is below it has run.
    synced: AtomicU64,
}

/// A copy in flight on a stream, holding its host end until a sync of the
/// stream has covered the copy: what [`Stream::upload`] and
/// [`Stream::download`] return. The host end is moved in, so nothing can
/// read or write it while the copy may still run:
///
/// ```compile_fail,E0382
/// use fb_core_device::{DeviceSliceMut, HostSliceMut, Stream};
///
/// fn rewrite(
///     stream: &Stream,
///     dst: &mut impl DeviceSliceMut<f64>,
///     mut staged: impl HostSliceMut<f64>,
/// ) {
///     let copy = stream.upload(dst, staged).unwrap();
///     let _ = staged.as_mut_ptr(); // the host end is the token's
///     let _ = copy.sync();
/// }
/// ```
///
/// [`Pending::sync`] hands it back. Several copies on one stream share one
/// sync: a token whose copy an earlier [`Stream::sync`] or
/// [`Pending::sync`] of the stream already covered returns its host end
/// without blocking again, so a step's downloads followed by one sync cost
/// that one sync:
///
/// ```no_run
/// use fb_core_device::{DeviceSlice, HostSliceMut, Stream};
/// use fb_core_error_handling::Result;
///
/// fn both<H: HostSliceMut<f64>>(
///     stream: &Stream,
///     (a, b): (H, H),
///     (d_a, d_b): (&impl DeviceSlice<f64>, &impl DeviceSlice<f64>),
/// ) -> Result<(H, H)> {
///     let a = stream.download(a, d_a)?;
///     let b = stream.download(b, d_b)?;
///     let b = b.sync()?; // the one sync
///     Ok((a.sync()?, b)) // covered by it
/// }
/// ```
///
/// One host end goes to several streams with one token, the same bytes to
/// each GPU: [`Pending::and_upload`] enqueues a second upload of it and
/// joins that copy to the token's stream through an event, so the token's
/// one sync covers both:
///
/// ```no_run
/// use fb_core_device::{DeviceSliceMut, HostSlice, Stream};
/// use fb_core_error_handling::Status;
///
/// fn broadcast(
///     (gpu0, gpu1): (&Stream, &Stream),
///     (d0, d1): (&mut impl DeviceSliceMut<f64>, &mut impl DeviceSliceMut<f64>),
///     staged: impl HostSlice<f64>,
/// ) -> Status {
///     let staged = gpu0.upload(d0, staged)?.and_upload(gpu1, d1)?.sync()?;
///     drop(staged); // both copies have run
///     Ok(())
/// }
/// ```
///
/// Only an upload's token takes a second copy: a download writes its host
/// end, so nothing may read it until the download has run. Its token is a
/// `Pending<'_, H, false>`, which has no [`and_upload`](Pending::and_upload):
///
/// ```compile_fail,E0599
/// use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut, Stream};
///
/// fn read_while_written<H: HostSlice<f64> + HostSliceMut<f64>>(
///     stream: &Stream,
///     back: H,
///     src: &impl DeviceSlice<f64>,
///     dst: &mut impl DeviceSliceMut<f64>,
/// ) {
///     let copy = stream.download(back, src).unwrap();
///     let _ = copy.and_upload(stream, dst); // the download may be writing it
/// }
/// ```
///
/// A ring rewrites a buffer's host end once the copy that read it has run:
/// each buffer goes through a stream of its own, so the token's sync waits
/// for that buffer's last copy and no other's.
///
/// Dropped unsynced, a token leaks its host end, never freeing or handing
/// back memory a copy may still use. A lease's tenant costs nothing that
/// way (its bytes are the step's until the step-end sync); an owned buffer
/// stays allocated, so sync it to have it back.
///
/// `UPLOAD` says the copies only read the host end: true for an upload's
/// token, false for a download's.
#[must_use = "dropped unsynced, the token leaks its host end; `sync` hands it back"]
pub struct Pending<'s, H, const UPLOAD: bool = true> {
    stream: &'s Stream,
    ticket: u64,
    host: ManuallyDrop<H>,
}

impl<'s, H> Pending<'s, H> {
    /// Enqueue on `stream` the upload of the head of the host end, its
    /// first `dst.len()` elements, into device `dst`, and join it to the
    /// token's stream: an event recorded on `stream` after the copy, which
    /// the token's stream waits for. The token returned covers both copies,
    /// and every copy it covered before, so its sync, or a sync of its
    /// stream begun after this, waits for them all. `stream` may be another
    /// device's; the token's stream then waits for that device's copy
    /// before anything enqueued on it after this call.
    ///
    /// # Errors
    /// The failure of the enqueue, the event's record or the wait; the
    /// host end is then leaked, since a copy of it may still run.
    #[requires(le!(dst.len(), self.host.len()), "the host end holds the device end")]
    #[track_caller]
    pub fn and_upload<T: Copy>(
        self,
        stream: &Stream,
        dst: &mut impl DeviceSliceMut<T>,
    ) -> Result<Pending<'s, H>>
    where
        H: HostSlice<T>,
    {
        // No overflow: the device slice promises `len()` elements of memory.
        let bytes = dst.len() * size_of::<T>();
        // SAFETY: the host end holds at least `bytes` of host memory that
        // only the value reaches (`HostSlice`'s promise); every copy the
        // token covers only reads it (`UPLOAD`), and the token returned
        // keeps it, unread and unwritten, until a sync covers this copy too,
        // or leaks it, as does every error return here (the token has no
        // drop); `dst` is `bytes` of device memory with no other user until
        // the work enqueued here finishes (`DeviceSliceMut`'s).
        unsafe {
            stream.copy(
                dst.as_mut_ptr().cast(),
                self.host.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            )
        }?;
        // Destroyed at the end of the call: the wait holds the point it
        // was recorded at, not the event.
        let joined = Event::new(stream.physical);
        joined.record(stream)?;
        self.stream.wait(&joined)?;
        Ok(self.stream.pending(ManuallyDrop::into_inner(self.host)))
    }
}

impl<H, const UPLOAD: bool> Pending<'_, H, UPLOAD> {
    /// The host end, once its copy has run: no wait when a sync of the
    /// stream already covered the copy, else a [`Stream::sync`].
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced by the sync; the host end is
    /// then leaked, since the copy may not have run.
    #[track_caller]
    pub fn sync(self) -> Result<H> {
        let Pending {
            stream,
            ticket,
            host,
        } = self;
        if stream.synced.load(Ordering::SeqCst) <= ticket {
            #[allow(
                clippy::disallowed_methods,
                reason = "the wrapper the lint names: a token's sync is the stream's, and its callers carry the allow"
            )]
            stream.sync()?;
        }
        Ok(ManuallyDrop::into_inner(host))
    }
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
        Stream {
            raw,
            physical,
            tickets: AtomicU64::new(0),
            synced: AtomicU64::new(0),
        }
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
    /// [`sync_meter`]. Every [`Pending`] of a copy enqueued before it
    /// returns its host end after it without blocking again.
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced here.
    #[track_caller]
    pub fn sync(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // Taken before the runtime's sync: a token whose ticket was read
        // before this, `ticket` or less, was enqueued before the sync began
        // (one total order of the two atomics' sequentially consistent
        // operations).
        let ticket = self.tickets.fetch_add(1, Ordering::SeqCst);
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
        runtime_status(status)?;
        self.synced.fetch_max(ticket + 1, Ordering::SeqCst);
        Ok(())
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

    /// Enqueue the copy of the head of `src`, its first `dst.len()`
    /// elements, into device `dst`. The token holds `src` until a sync of
    /// the stream has covered the copy.
    ///
    /// # Errors
    /// The enqueue's failure; nothing was enqueued, and `src` is dropped.
    #[requires(le!(dst.len(), src.len()), "the host end holds the device end")]
    #[track_caller]
    pub fn upload<T: Copy, H: HostSlice<T>>(
        &self,
        dst: &mut impl DeviceSliceMut<T>,
        src: H,
    ) -> Result<Pending<'_, H>> {
        // No overflow: the device slice promises `len()` elements of memory.
        let bytes = dst.len() * size_of::<T>();
        // SAFETY: `src` holds at least `bytes` of host memory that only the
        // value reaches, and the token keeps the value, unread and unwritten,
        // until a sync covers the copy, or leaks it (`HostSlice`'s promise);
        // `dst` is `bytes` of device memory with no other user until the
        // work enqueued here finishes (`DeviceSliceMut`'s).
        unsafe {
            self.copy(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            )
        }?;
        Ok(self.pending(src))
    }

    /// Enqueue the copy of device `src` into the head of `dst`, its first
    /// `src.len()` elements; the rest of `dst` is left as it is. The token
    /// holds `dst` until a sync of the stream has covered the copy, and
    /// hands it back to be read.
    ///
    /// # Errors
    /// The enqueue's failure; nothing was enqueued, and `dst` is dropped.
    #[requires(le!(src.len(), dst.len()), "the host end holds the device end")]
    #[track_caller]
    pub fn download<T: Copy, H: HostSliceMut<T>>(
        &self,
        mut dst: H,
        src: &impl DeviceSlice<T>,
    ) -> Result<Pending<'_, H, false>> {
        // No overflow: the device slice promises `len()` elements of memory.
        let bytes = src.len() * size_of::<T>();
        // SAFETY: `dst` holds at least `bytes` of host memory that only the
        // value reaches, and the token keeps the value, unread and unwritten,
        // until a sync covers the copy, or leaks it (`HostSliceMut`'s
        // promise); `src` is `bytes` of device memory nothing writes until
        // the work enqueued here finishes (`DeviceSlice`'s).
        unsafe {
            self.copy(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                bytes,
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
            )
        }?;
        Ok(self.pending(dst))
    }

    /// The token of a copy just enqueued, holding its host end.
    fn pending<H, const UPLOAD: bool>(&self, host: H) -> Pending<'_, H, UPLOAD> {
        Pending {
            stream: self,
            // Read after the enqueue: a sync whose ticket is this or more
            // began after the copy was enqueued.
            ticket: self.tickets.load(Ordering::SeqCst),
            host: ManuallyDrop::new(host),
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

    /// The one `wwr_memcpy_async` of the three copies; nothing is enqueued
    /// for no bytes.
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
            reason = "the host's wait on one point of a stream, not all of it (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
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
