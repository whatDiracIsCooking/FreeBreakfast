//! `wwr.extension.runtime:event`: the owning event and the borrow-safe
//! event operations.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_event, wwr_status, wwr_stream};

use super::first_failure;
use super::stream::Stream;
use crate::core::error_handling::{Result, check_or_abort};
use crate::core::wwr::common::SourceLocation;
use crate::core::wwr::handle::{DeviceBoundHandle, DeviceHandle, DeviceHandleStream, HandleKind};

/// `cudaEventBlockingSync` / `hipEventBlockingSync`: a host wait on the
/// event blocks instead of spinning.
pub const BLOCKING_SYNC: u32 = 0x1;

/// `cudaEventDisableTiming` / `hipEventDisableTiming`: the event records no
/// timestamp, so it is sync-only and cheaper.
pub const DISABLE_TIMING: u32 = 0x2;

/// An event's create and destroy. The default create disables timing, the
/// common and cheaper case; the destroy synchronizes first, as wwr's does.
pub struct EventKind;

impl HandleKind for EventKind {
    type Raw = wwr_event;
    type Error = wwr_status;

    unsafe fn create(raw: *mut wwr_event) -> wwr_status {
        // SAFETY: the caller hands a pointer valid for one write.
        unsafe { wwr_sys::wwr_event_create_with_flags(raw, DISABLE_TIMING) }
    }

    unsafe fn destroy(raw: wwr_event) -> wwr_status {
        // SAFETY: `raw` is a live event, destroyed once, after it completes.
        unsafe {
            first_failure(
                wwr_sys::wwr_event_synchronize(raw),
                wwr_sys::wwr_event_destroy(raw),
            )
        }
    }
}

/// An owned event on one device: `EventWrapper`. Every constructor aborts
/// on failure ([`super`] has why).
pub struct Event(DeviceBoundHandle<EventKind>);

impl Event {
    /// A timing-disabled event on `dev_idx`.
    #[track_caller]
    pub fn new(dev_idx: i32) -> Self {
        check_or_abort(Self::try_with_flags_at(
            dev_idx,
            DISABLE_TIMING,
            Location::caller(),
        ))
    }

    /// An event on `dev_idx` with `flags` (`wwrEventCreateWithFlags`); pass
    /// `0` for a timing-capable event.
    #[track_caller]
    pub fn with_flags(dev_idx: i32, flags: u32) -> Self {
        check_or_abort(Self::try_with_flags_at(dev_idx, flags, Location::caller()))
    }

    /// The fallible constructor every public one aborts over.
    fn try_with_flags_at(dev_idx: i32, flags: u32, location: SourceLocation) -> Result<Self> {
        DeviceBoundHandle::try_new_at(
            dev_idx,
            // SAFETY: `raw` is valid for one write.
            |raw| unsafe { wwr_sys::wwr_event_create_with_flags(raw, flags) },
            location,
        )
        .map(Event)
    }

    /// [`record`] this event on `stream`.
    pub fn record(&self, stream: &Stream) -> wwr_status {
        self.record_with_flags(stream, 0)
    }

    /// [`record_with_flags`] this event on `stream`.
    pub fn record_with_flags(&self, stream: &Stream, flags: u32) -> wwr_status {
        // SAFETY: both borrows keep their handles alive.
        unsafe { record_with_flags(self.get(), stream.stream(), flags) }
    }

    /// [`sync`] on this event.
    pub fn sync(&self) -> wwr_status {
        // SAFETY: the borrow keeps the event alive.
        unsafe { sync(self.get()) }
    }
}

impl Deref for Event {
    type Target = DeviceBoundHandle<EventKind>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeviceHandle for Event {
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

// The borrow-safe operations on a raw event, `unsafe` for the reason
// `stream`'s are.

/// Record `event` at the current end of `stream`'s queue.
///
/// # Safety
/// `event` is a live event, and `stream` a live stream or null (the legacy
/// default stream).
pub unsafe fn record(event: wwr_event, stream: wwr_stream) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_event_record(event, stream) }
}

/// [`record`], with `wwrEventRecordWithFlags`'s flags.
///
/// # Safety
/// As [`record`].
pub unsafe fn record_with_flags(event: wwr_event, stream: wwr_stream, flags: u32) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_event_record_with_flags(event, stream, flags) }
}

/// Wait for the work `event` was recorded after to finish.
///
/// # Safety
/// `event` is a live event.
pub unsafe fn sync(event: wwr_event) -> wwr_status {
    // SAFETY: the caller's.
    unsafe { wwr_sys::wwr_event_synchronize(event) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error_handling::ErrorKind;
    use crate::core::wwr::runtime::tests::{DeviceBytes, ok};

    /// Needs a GPU. The default event cannot time; a `0`-flag one can.
    #[test]
    fn timing_is_off_by_default_and_on_with_zero_flags() {
        let s = Stream::new(0);
        let sync_only = Event::new(0);
        let (start, end) = (Event::with_flags(0, 0), Event::with_flags(0, 0));
        assert_eq!(sync_only.dev_idx(), 0);

        ok(start.record(&s));
        ok(sync_only.record(&s));
        ok(end.record(&s));
        ok(end.sync());

        let mut ms = -1.0f32;
        // SAFETY: writes one f32 through a valid local.
        ok(unsafe { wwr_sys::wwr_event_elapsed_time(&mut ms, start.get(), end.get()) });
        assert!(ms >= 0.0);
        // SAFETY: as above.
        let untimed =
            unsafe { wwr_sys::wwr_event_elapsed_time(&mut ms, start.get(), sync_only.get()) };
        assert_ne!(untimed, wwr_status::WWR_STATUS_SUCCESS);
    }

    /// Needs a GPU. A stream that waits on an event recorded on another
    /// runs its work after the other's: the second memset wins.
    #[test]
    fn wait_event_orders_two_streams() {
        let (a, b) = (Stream::new(0), Stream::new(0));
        let done = Event::new(0);
        let bytes = DeviceBytes::new(1 << 20);
        // SAFETY: the allocation's length, queued on live streams.
        ok(unsafe { wwr_sys::wwr_memset_async(bytes.0, 1, 1 << 20, a.stream()) });
        ok(done.record(&a));
        ok(b.wait_event(&done, 0));
        // SAFETY: as above.
        ok(unsafe { wwr_sys::wwr_memset_async(bytes.0, 2, 1 << 20, b.stream()) });
        ok(a.sync());
        assert!(bytes.read(b.stream()).iter().all(|&x| x == 2));
    }

    /// Needs a GPU. Flags no backend accepts fail the create, at the
    /// caller, and no event is made.
    #[test]
    fn bad_flags_are_an_error_at_the_caller() {
        let line = line!() + 1;
        let error = Event::try_with_flags_at(0, u32::MAX, Location::caller())
            .err()
            .unwrap();
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
    }
}
