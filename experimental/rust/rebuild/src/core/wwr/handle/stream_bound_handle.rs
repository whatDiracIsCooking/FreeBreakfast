//! `wwr.extension.handle:stream_bound_handle`.

use std::ops::Deref;
use std::panic::Location;
use std::rc::Rc;

use wwr_sys::{wwr_status, wwr_stream};

use super::base_handle::create_raw;
use super::{
    DeviceBoundHandle, DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandleView,
};
use crate::core::wwr::common::{ErrorPolicy, gpu_check_at};

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
/// owner's device, the owner's stream is bound to it through the create
/// policy (`...SetStream` returns the kind's own status), and the owner is
/// retained so the stream outlives the handle. [`DeviceHandleStream::stream`]
/// reads back through the owner, so there is one source of truth.
///
/// The owner is an `Rc`, wwr's `shared_ptr` without the atomic count: a
/// handle over a raw GPU pointer is neither `Send` nor `Sync`, so the count
/// is never shared across threads. Ownership runs one way only, handle to
/// owner. An owner that came to own a handle anchored back to it would be an
/// `Rc` cycle and leak, which wwr documents and this cannot prevent either.
///
/// Fields drop in declaration order, so the handle is destroyed before the
/// owner it was bound to is released.
///
/// wwr binds the stream even when the create failed and there is no handle
/// to bind it to; this binds only a live handle.
pub struct StreamBoundHandle<K, PC, PD, S, PDev>
where
    K: StreamBoundKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    S: DeviceHandleStream,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    handle: DeviceBoundHandle<K, PC, PD, PDev>,
    owner: Rc<S>,
}

impl<K, PC, PD, S, PDev> StreamBoundHandle<K, PC, PD, S, PDev>
where
    K: StreamBoundKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    S: DeviceHandleStream,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// Create the handle on `owner`'s device and bind `owner`'s stream to it,
    /// failures reported at the caller's location.
    #[track_caller]
    pub fn new(owner: Rc<S>, policy_create: PC, policy_destroy: PD, policy_device: PDev) -> Self {
        let location = Location::caller();
        let mut handle = DeviceBoundHandle::new_at(
            owner.dev_idx(),
            policy_create,
            policy_destroy,
            policy_device,
            create_raw::<K, PC>,
            location,
        );
        if let Some(raw) = handle.base.raw {
            // SAFETY: `raw` was just created and is live; the stream is
            // live for as long as `owner` is, which this handle retains.
            gpu_check_at(
                unsafe { K::set_stream(raw, owner.stream()) },
                &mut handle.base.policy_create,
                location,
            );
        }
        StreamBoundHandle { handle, owner }
    }

    /// The retained stream owner.
    pub fn owner(&self) -> &Rc<S> {
        &self.owner
    }

    /// A copyable view carrying the device and the bound stream, which
    /// cannot outlive `self`; `None` when there is no live handle to view.
    pub fn view(&self) -> Option<StreamBoundHandleView<'_, K::Raw>> {
        self.handle
            .get()
            .map(|raw| StreamBoundHandleView::from_raw(raw, self.dev_idx(), self.stream()))
    }
}

impl<K, PC, PD, S, PDev> DeviceHandle for StreamBoundHandle<K, PC, PD, S, PDev>
where
    K: StreamBoundKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    S: DeviceHandleStream,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    fn dev_idx(&self) -> i32 {
        self.handle.dev_idx()
    }
}

impl<K, PC, PD, S, PDev> DeviceHandleStream for StreamBoundHandle<K, PC, PD, S, PDev>
where
    K: StreamBoundKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    S: DeviceHandleStream,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The bound stream, read back through the retained owner.
    fn stream(&self) -> wwr_stream {
        self.owner.stream()
    }
}

impl<K, PC, PD, S, PDev> Deref for StreamBoundHandle<K, PC, PD, S, PDev>
where
    K: StreamBoundKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    S: DeviceHandleStream,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    type Target = DeviceBoundHandle<K, PC, PD, PDev>;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{BoundEventKind, OwnedStream};
    use super::*;
    use crate::core::wwr::common::tests::Recording;

    type Bound = StreamBoundHandle<BoundEventKind, Recording, Recording, OwnedStream, Recording>;

    /// Needs a GPU. The handle lands on the owner's device, binds the
    /// owner's stream (here: records the event on it), keeps the owner alive
    /// while it lives, and lets go when it drops.
    #[test]
    fn binds_and_retains_the_owners_stream() {
        let owner = Rc::new(OwnedStream::new(0));
        let h = Bound::new(
            owner.clone(),
            Recording::default(),
            Recording::default(),
            Recording::default(),
        );
        assert!(h.get().is_some());
        assert_eq!(h.create_policy().0, []);
        assert_eq!(h.device_policy().0, []);
        assert_eq!(h.dev_idx(), owner.dev_idx());
        assert_eq!(h.stream(), owner.stream());
        assert_eq!(Rc::strong_count(&owner), 2);

        let v = h.view().unwrap();
        assert_eq!(
            (v.get(), v.dev_idx(), v.stream()),
            (h.get().unwrap(), 0, owner.stream())
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
