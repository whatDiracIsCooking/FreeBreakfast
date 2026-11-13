//! `wwr.extension.handle:stream_bound_handle`.

use std::ops::Deref;
use std::panic::Location;
use std::rc::Rc;

use wwr_sys::wwr_stream;

use super::{
    DeviceBoundHandle, DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandleView,
};
use crate::gpu_status_at;
use fb_core_error_handling::Result;

/// A [`HandleKind`] that binds a work stream after it is created: a BLAS,
/// solver, sparse or FFT handle, through its vendor `...SetStream`. This is
/// the `set_stream` hook wwr's `StreamBoundHandle` calls on its `Derived`.
pub trait StreamBoundKind: HandleKind {
    /// Bind `stream` to `raw`.
    ///
    /// # Safety
    /// `raw` is a live handle of this kind and `stream` a live stream.
    unsafe fn set_stream(raw: Self::Raw, stream: wwr_stream) -> Self::Error;
}

/// A [`DeviceBoundHandle`] bound to a stream it keeps alive:
/// `StreamBoundHandle`.
///
/// It is made from a shared stream owner `S`. The handle is created on the
/// owner's device, the owner's stream is bound to it (`...SetStream`
/// returns the kind's own status; a failure is an `Err`, and the handle is
/// destroyed), and the owner is retained so the stream outlives the handle.
/// [`DeviceHandleStream::stream`] reads back through the owner, so there is
/// one source of truth.
///
/// The owner is an `Rc`, wwr's `shared_ptr` without the atomic count: a
/// handle over a raw GPU pointer is neither `Send` nor `Sync`, so the count
/// is never shared across threads. Ownership runs one way only, handle to
/// owner. An owner that came to own a handle anchored back to it would be an
/// `Rc` cycle and leak, which wwr documents and this cannot prevent either.
///
/// Fields drop in declaration order, so the handle is destroyed before the
/// owner it was bound to is released.
pub struct StreamBoundHandle<K: StreamBoundKind, S: DeviceHandleStream> {
    handle: DeviceBoundHandle<K>,
    owner: Rc<S>,
}

impl<K: StreamBoundKind, S: DeviceHandleStream> StreamBoundHandle<K, S> {
    /// Create the handle on `owner`'s device and bind `owner`'s stream to
    /// it; a failure is an `Err` at the caller's location.
    #[track_caller]
    pub fn try_new(owner: Rc<S>) -> Result<Self> {
        let location = Location::caller();
        let handle = DeviceBoundHandle::try_new_at(
            owner.dev_idx(),
            // SAFETY: `try_new_at` hands a pointer valid for one write.
            |raw| unsafe { K::create(raw) },
            location,
        )?;
        // SAFETY: the handle was just created and is live; the stream is
        // live for as long as `owner` is, which this handle retains.
        gpu_status_at(
            unsafe { K::set_stream(handle.get(), owner.stream()) },
            location,
        )?;
        Ok(StreamBoundHandle { handle, owner })
    }

    /// The retained stream owner.
    pub fn owner(&self) -> &Rc<S> {
        &self.owner
    }

    /// A copyable view carrying the device and the bound stream, which
    /// cannot outlive `self`.
    pub fn view(&self) -> StreamBoundHandleView<'_, K::Raw> {
        StreamBoundHandleView::from_raw(self.handle.get(), self.dev_idx(), self.stream())
    }
}

impl<K: StreamBoundKind, S: DeviceHandleStream> DeviceHandle for StreamBoundHandle<K, S> {
    fn dev_idx(&self) -> i32 {
        self.handle.dev_idx()
    }
}

impl<K: StreamBoundKind, S: DeviceHandleStream> DeviceHandleStream for StreamBoundHandle<K, S> {
    /// The bound stream, read back through the retained owner.
    fn stream(&self) -> wwr_stream {
        self.owner.stream()
    }
}

impl<K: StreamBoundKind, S: DeviceHandleStream> Deref for StreamBoundHandle<K, S> {
    type Target = DeviceBoundHandle<K>;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{BoundEventKind, OwnedStream};
    use super::*;
    use wwr_sys::wwr_status;

    type Bound = StreamBoundHandle<BoundEventKind, OwnedStream>;

    /// Needs a GPU. The handle lands on the owner's device, binds the
    /// owner's stream (here: records the event on it), keeps the owner alive
    /// while it lives, and lets go when it drops.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn binds_and_retains_the_owners_stream() {
        let owner = Rc::new(OwnedStream::new(0));
        let h = Bound::try_new(owner.clone()).unwrap();
        assert_eq!(h.dev_idx(), owner.dev_idx());
        assert_eq!(h.stream(), owner.stream());
        assert_eq!(Rc::strong_count(&owner), 2);

        let v = h.view();
        assert_eq!(
            (v.get(), v.dev_idx(), v.stream()),
            (h.get(), 0, owner.stream())
        );
        // SAFETY: the event is live and was recorded on the stream.
        assert_eq!(
            unsafe { wwr_sys::wwr_event_synchronize(v.get()) },
            wwr_status::WWR_STATUS_SUCCESS
        );

        drop(h);
        assert_eq!(Rc::strong_count(&owner), 1);
    }
}
