//! `wwr.extension.handle:device_bound_handle`.

use std::ops::Deref;
use std::panic::Location;

use wwr_sys::wwr_status;

use super::base_handle::create_raw;
use super::{BaseHandle, DeviceBoundHandleView, DeviceHandle, HandleKind};
use crate::core::wwr::common::{ErrorPolicy, SourceLocation, gpu_check_at};

/// A [`BaseHandle`] that records the device it was created on:
/// `DeviceBoundHandle`.
///
/// Every constructor names the device. It is made current through the
/// device policy, the handle is created there, and the index recorded is
/// then read back from the runtime, so it is the device the handle actually
/// landed on. The device policy handles `wwr_status` whatever the kind's own
/// status is, which is what lets a library handle (BLAS status, say) report
/// its device calls too.
///
/// It derefs to its [`BaseHandle`], which is how wwr's public inheritance
/// comes across: `get()` and the create/destroy policies are the base's,
/// and [`DeviceBoundHandle::view`] hides the base's view as wwr's does.
///
/// wwr's `select_device` / `record_device`, for a wrapper that creates its
/// own handle, are folded into [`DeviceBoundHandle::new_with`]: the select,
/// the given create and the record, in that order.
pub struct DeviceBoundHandle<K, PC, PD, PDev>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    pub(super) base: BaseHandle<K, PC, PD>,
    dev_idx: i32,
    policy_device: PDev,
}

impl<K, PC, PD, PDev> DeviceBoundHandle<K, PC, PD, PDev>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// Create a handle with `K::create` on device `dev_idx`, failures
    /// reported at the caller's location.
    #[track_caller]
    pub fn new(dev_idx: i32, policy_create: PC, policy_destroy: PD, policy_device: PDev) -> Self {
        Self::new_at(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
            create_raw::<K, PC>,
            Location::caller(),
        )
    }

    /// Create a handle with `create` on device `dev_idx`, as
    /// [`BaseHandle::new_with`].
    #[track_caller]
    pub fn new_with(
        dev_idx: i32,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
        create: impl FnOnce(&mut PC, SourceLocation) -> Option<K::Raw>,
    ) -> Self {
        Self::new_at(
            dev_idx,
            policy_create,
            policy_destroy,
            policy_device,
            create,
            Location::caller(),
        )
    }

    pub(super) fn new_at(
        dev_idx: i32,
        policy_create: PC,
        policy_destroy: PD,
        mut policy_device: PDev,
        create: impl FnOnce(&mut PC, SourceLocation) -> Option<K::Raw>,
        location: SourceLocation,
    ) -> Self {
        // SAFETY: a plain runtime call; a bad index is a status, not UB.
        gpu_check_at(
            unsafe { wwr_sys::wwr_set_device(dev_idx) },
            &mut policy_device,
            location,
        );
        let base = BaseHandle::new_at(policy_create, policy_destroy, create, location);
        let mut recorded = -1;
        // SAFETY: writes one i32 through a valid local.
        gpu_check_at(
            unsafe { wwr_sys::wwr_get_device(&mut recorded) },
            &mut policy_device,
            location,
        );
        DeviceBoundHandle {
            base,
            dev_idx: recorded,
            policy_device,
        }
    }

    /// The policy that handled the device calls, for reading back what a
    /// stateful one recorded.
    pub fn device_policy(&self) -> &PDev {
        &self.policy_device
    }

    /// The base's create policy. There is no `DerefMut` to the base: a
    /// `BaseHandle` assigned through it would carry this one's device index
    /// without having been created there.
    pub fn create_policy_mut(&mut self) -> &mut PC {
        self.base.create_policy_mut()
    }

    /// A copyable view carrying the device, which cannot outlive `self`;
    /// `None` when there is no live handle to view.
    pub fn view(&self) -> Option<DeviceBoundHandleView<'_, K::Raw>> {
        self.base
            .get()
            .map(|raw| DeviceBoundHandleView::from_raw(raw, self.dev_idx))
    }
}

impl<K, PC, PD, PDev> DeviceHandle for DeviceBoundHandle<K, PC, PD, PDev>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// The device the handle was created on (-1 if it could not be read).
    fn dev_idx(&self) -> i32 {
        self.dev_idx
    }
}

impl<K, PC, PD, PDev> Deref for DeviceBoundHandle<K, PC, PD, PDev>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    type Target = BaseHandle<K, PC, PD>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{StreamKind, current_device};
    use super::*;
    use crate::core::wwr::common::tests::Recording;

    type Stream = DeviceBoundHandle<StreamKind, Recording, Recording, Recording>;

    #[track_caller]
    fn stream(dev_idx: i32) -> Stream {
        Stream::new(
            dev_idx,
            Recording::default(),
            Recording::default(),
            Recording::default(),
        )
    }

    /// Needs a GPU.
    #[test]
    fn records_the_device_it_was_created_on() {
        let s = stream(0);
        assert!(s.get().is_some());
        assert_eq!(s.dev_idx(), 0);
        assert_eq!(s.device_policy().0, []);
        assert_eq!(s.create_policy().0, []);
        let v = s.view().unwrap();
        assert_eq!((v.get(), v.dev_idx()), (s.get().unwrap(), 0));
    }

    /// Needs a GPU. A bad index fails the select through the device policy;
    /// the handle lands on whatever device was current, and that is the one
    /// recorded.
    #[test]
    fn a_bad_index_is_reported_and_the_actual_device_recorded() {
        let current = current_device();
        let line = line!() + 1;
        let s = stream(-1);
        assert_eq!(
            s.device_policy().0,
            [(wwr_status::WWR_STATUS_INVALID_DEVICE, line)]
        );
        assert!(s.get().is_some());
        assert_eq!(s.dev_idx(), current);
    }
}
