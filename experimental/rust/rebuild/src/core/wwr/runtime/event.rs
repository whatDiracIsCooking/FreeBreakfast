//! `wwr.extension.runtime:event`: the owning event and the borrow-safe
//! event operations.

use std::ops::Deref;

use wwr_sys::{wwr_event, wwr_status, wwr_stream};

use super::stream::Stream;
use super::{create_with, first_failure};
use crate::core::wwr::common::ErrorPolicy;
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

/// An owned event on one device: `EventWrapper<P_create, P_destroy,
/// P_device>`.
pub struct Event<PC, PD, PDev>(DeviceBoundHandle<EventKind, PC, PD, PDev>)
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>;

impl<PC, PD, PDev> Event<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// A timing-disabled event on `dev_idx`.
    #[track_caller]
    pub fn new(dev_idx: i32, policy_create: PC, policy_destroy: PD, policy_device: PDev) -> Self {
        Event(DeviceBoundHandle::new(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
        ))
    }

    /// An event on `dev_idx` with `flags` (`wwrEventCreateWithFlags`); pass
    /// `0` for a timing-capable event.
    #[track_caller]
    pub fn with_flags(
        dev_idx: i32,
        flags: u32,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
    ) -> Self {
        Event(DeviceBoundHandle::new_with(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
            |policy, location| {
                create_with(policy, location, |raw| {
                    // SAFETY: `raw` is valid for one write.
                    unsafe { wwr_sys::wwr_event_create_with_flags(raw, flags) }
                })
            },
        ))
    }
}

impl<PC, PD, PDev> Deref for Event<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    type Target = DeviceBoundHandle<EventKind, PC, PD, PDev>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<PC, PD, PDev> DeviceHandle for Event<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

impl<PC, PD, PDev> Event<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// [`record`] this event on `stream`.
    pub fn record<SC, SD, SDev>(&self, stream: &Stream<SC, SD, SDev>) -> wwr_status
    where
        SC: ErrorPolicy<Error = wwr_status>,
        SD: ErrorPolicy<Error = wwr_status>,
        SDev: ErrorPolicy<Error = wwr_status>,
    {
        self.record_with_flags(stream, 0)
    }

    /// [`record_with_flags`] this event on `stream`;
    /// `WWR_STATUS_INVALID_VALUE` if the event owns nothing.
    pub fn record_with_flags<SC, SD, SDev>(
        &self,
        stream: &Stream<SC, SD, SDev>,
        flags: u32,
    ) -> wwr_status
    where
        SC: ErrorPolicy<Error = wwr_status>,
        SD: ErrorPolicy<Error = wwr_status>,
        SDev: ErrorPolicy<Error = wwr_status>,
    {
        match self.get() {
            // SAFETY: both borrows keep their handles alive.
            Some(raw) => unsafe { record_with_flags(raw, stream.stream(), flags) },
            None => wwr_status::WWR_STATUS_INVALID_VALUE,
        }
    }

    /// [`sync`] on this event; `WWR_STATUS_INVALID_VALUE` if it owns
    /// nothing.
    pub fn sync(&self) -> wwr_status {
        match self.get() {
            // SAFETY: the borrow keeps the event alive.
            Some(raw) => unsafe { sync(raw) },
            None => wwr_status::WWR_STATUS_INVALID_VALUE,
        }
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
    use crate::core::wwr::handle::DeviceHandleStream;
    use crate::core::wwr::runtime::tests::{Abort, DeviceBytes, ok};

    /// Needs a GPU. The default event cannot time; a `0`-flag one can.
    #[test]
    fn timing_is_off_by_default_and_on_with_zero_flags() {
        let s = Stream::new(0, Abort::new(), Abort::new(), Abort::new());
        let sync_only = Event::new(0, Abort::new(), Abort::new(), Abort::new());
        let (start, end) = (
            Event::with_flags(0, 0, Abort::new(), Abort::new(), Abort::new()),
            Event::with_flags(0, 0, Abort::new(), Abort::new(), Abort::new()),
        );
        assert_eq!(sync_only.dev_idx(), 0);

        ok(start.record(&s));
        ok(sync_only.record(&s));
        ok(end.record(&s));
        ok(end.sync());

        let mut ms = -1.0f32;
        // SAFETY: writes one f32 through a valid local.
        ok(unsafe {
            wwr_sys::wwr_event_elapsed_time(&mut ms, start.get().unwrap(), end.get().unwrap())
        });
        assert!(ms >= 0.0);
        // SAFETY: as above.
        let untimed = unsafe {
            wwr_sys::wwr_event_elapsed_time(&mut ms, start.get().unwrap(), sync_only.get().unwrap())
        };
        assert_ne!(untimed, wwr_status::WWR_STATUS_SUCCESS);
    }

    /// Needs a GPU. A stream that waits on an event recorded on another
    /// runs its work after the other's: the second memset wins.
    #[test]
    fn wait_event_orders_two_streams() {
        let (a, b) = (
            Stream::new(0, Abort::new(), Abort::new(), Abort::new()),
            Stream::new(0, Abort::new(), Abort::new(), Abort::new()),
        );
        let done = Event::new(0, Abort::new(), Abort::new(), Abort::new());
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

    /// An event that owns nothing is refused before the runtime sees it.
    /// Needs a GPU for the stream.
    #[test]
    fn a_dead_event_is_refused() {
        use crate::core::wwr::common::tests::Recording;
        // Flags no backend accepts fail the create, under a policy that
        // returns, so the event owns nothing.
        let dead = Event::with_flags(
            0,
            u32::MAX,
            Recording::default(),
            Recording::default(),
            Recording::default(),
        );
        assert_eq!(dead.get(), None);
        let s = Stream::new(0, Abort::new(), Abort::new(), Abort::new());
        assert_eq!(dead.record(&s), wwr_status::WWR_STATUS_INVALID_VALUE);
        assert_eq!(dead.sync(), wwr_status::WWR_STATUS_INVALID_VALUE);
        assert_eq!(s.wait_event(&dead, 0), wwr_status::WWR_STATUS_INVALID_VALUE);
    }
}
