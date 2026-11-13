//! `wwr.extension.handle:device_bound_handle`.

use std::ops::Deref;
use std::panic::Location;

use super::{BaseHandle, DeviceBoundHandleView, DeviceHandle, HandleKind};
use crate::core::error_handling::Result;
use crate::core::wwr::common::SourceLocation;
use crate::core::wwr::gpu_status_at;

/// A [`BaseHandle`] that records the device it was created on:
/// `DeviceBoundHandle`.
///
/// Every constructor names the device. It is made current, the handle is
/// created there, and the index recorded is then read back from the
/// runtime, so it is the device the handle actually landed on. The select
/// and the read-back return `wwr_status` whatever the kind's own status is,
/// which is what lets a library handle (BLAS status, say) report its device
/// calls too. Any of the three failing is an `Err`, and no handle is made
/// (or, if only the read-back failed, the one made is destroyed).
///
/// It derefs to its [`BaseHandle`], which is how wwr's public inheritance
/// comes across: `get()` is the base's, and [`DeviceBoundHandle::view`]
/// hides the base's view as wwr's does.
///
/// wwr's `select_device` / `record_device`, for a wrapper that creates its
/// own handle, are folded into [`DeviceBoundHandle::try_new_with`]: the
/// select, the given create and the record, in that order.
pub struct DeviceBoundHandle<K: HandleKind> {
    base: BaseHandle<K>,
    dev_idx: i32,
}

impl<K: HandleKind> DeviceBoundHandle<K> {
    /// Create a handle with `K::create` on device `dev_idx`; a failure is an
    /// `Err` at the caller's location.
    #[track_caller]
    pub fn try_new(dev_idx: i32) -> Result<Self> {
        // SAFETY: `try_new_at` hands a pointer valid for one write.
        Self::try_new_at(dev_idx, |raw| unsafe { K::create(raw) }, Location::caller())
    }

    /// Create a handle with `create` on device `dev_idx`, as
    /// [`BaseHandle::try_new_with`].
    #[track_caller]
    pub fn try_new_with(
        dev_idx: i32,
        create: impl FnOnce(*mut K::Raw) -> K::Error,
    ) -> Result<Self> {
        Self::try_new_at(dev_idx, create, Location::caller())
    }

    pub(crate) fn try_new_at(
        dev_idx: i32,
        create: impl FnOnce(*mut K::Raw) -> K::Error,
        location: SourceLocation,
    ) -> Result<Self> {
        // SAFETY: a plain runtime call; a bad index is a status, not UB.
        gpu_status_at(unsafe { wwr_sys::wwr_set_device(dev_idx) }, location)?;
        let base = BaseHandle::try_new_at(create, location)?;
        let mut recorded = -1;
        // SAFETY: writes one i32 through a valid local.
        gpu_status_at(unsafe { wwr_sys::wwr_get_device(&mut recorded) }, location)?;
        Ok(DeviceBoundHandle {
            base,
            dev_idx: recorded,
        })
    }

    /// A copyable view carrying the device, which cannot outlive `self`.
    pub fn view(&self) -> DeviceBoundHandleView<'_, K::Raw> {
        DeviceBoundHandleView::from_raw(self.base.get(), self.dev_idx)
    }
}

impl<K: HandleKind> DeviceHandle for DeviceBoundHandle<K> {
    /// The device the handle was created on.
    fn dev_idx(&self) -> i32 {
        self.dev_idx
    }
}

impl<K: HandleKind> Deref for DeviceBoundHandle<K> {
    type Target = BaseHandle<K>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{StreamKind, current_device};
    use super::*;
    use crate::core::error_handling::ErrorKind;

    type Stream = DeviceBoundHandle<StreamKind>;

    /// Needs a GPU.
    #[test]
    fn records_the_device_it_was_created_on() {
        let s = Stream::try_new(0).unwrap();
        assert_eq!(s.dev_idx(), 0);
        let v = s.view();
        assert_eq!((v.get(), v.dev_idx()), (s.get(), 0));
    }

    /// Needs a GPU. A bad index fails the select, at the caller; no handle
    /// is made, and the current device is left as it was.
    #[test]
    fn a_bad_index_is_an_error_at_the_caller() {
        let current = current_device();
        let line = line!() + 1;
        let error = Stream::try_new(-1).err().unwrap();
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert!(error.message.starts_with("WWR_STATUS_INVALID_DEVICE"));
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
        assert_eq!(current_device(), current);
    }
}
