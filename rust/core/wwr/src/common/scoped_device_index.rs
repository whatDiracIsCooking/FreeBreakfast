//! `wwr.extension.common:scoped_device_index`.

use std::marker::PhantomData;
use std::panic::Location;

use super::SourceLocation;
use crate::gpu_status_at;
use fb_core_error_handling::{Result, check_or_abort};

/// Make a device current for the guard's lifetime, and restore the one that
/// was current before when it drops.
///
/// [`ScopedDeviceIndex::new`] aborts if it cannot read or set the device,
/// as every create does; [`ScopedDeviceIndex::try_new`] returns the `Err`
/// instead. The restore on drop **aborts** on failure too, unlike a
/// handle's destroy, which logs: a thread left on the wrong device sends
/// every later launch and allocation to the wrong card, which is worse than
/// stopping.
///
/// The C++ guard cannot be moved. This one can, harmlessly: the current
/// device is a property of the thread, not of the guard's address. It is
/// neither `Send` nor `Sync`, though, because restoring on another thread
/// would set that thread's device and leave this one's switched.
pub struct ScopedDeviceIndex {
    /// The device current at construction, restored on drop.
    pub original_idx: i32,
    not_send: PhantomData<*const ()>,
}

impl ScopedDeviceIndex {
    /// Switch to `target_idx`, recording the current device to restore;
    /// abort at the caller if either call fails.
    #[track_caller]
    pub fn new(target_idx: i32) -> Self {
        check_or_abort(Self::try_new_at(target_idx, Location::caller()))
    }

    /// [`ScopedDeviceIndex::new`], with a failure as an `Err` at the caller.
    /// Nothing was switched when it fails.
    #[track_caller]
    pub fn try_new(target_idx: i32) -> Result<Self> {
        Self::try_new_at(target_idx, Location::caller())
    }

    fn try_new_at(target_idx: i32, location: SourceLocation) -> Result<Self> {
        let mut original_idx = -1;
        // SAFETY: a plain runtime call writing one i32 through a valid local.
        gpu_status_at(
            unsafe { wwr_sys::wwr_get_device(&mut original_idx) },
            location,
        )?;
        // SAFETY: a plain runtime call; a bad index is a status, not UB.
        gpu_status_at(unsafe { wwr_sys::wwr_set_device(target_idx) }, location)?;
        Ok(ScopedDeviceIndex {
            original_idx,
            not_send: PhantomData,
        })
    }
}

impl Drop for ScopedDeviceIndex {
    fn drop(&mut self) {
        // SAFETY: as in try_new_at. The location reported is this line, as
        // wwr's destructor reports its own.
        check_or_abort(gpu_status_at(
            unsafe { wwr_sys::wwr_set_device(self.original_idx) },
            Location::caller(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fb_core_error_handling::ErrorKind;
    use wwr_sys::wwr_status;

    fn current() -> i32 {
        let mut d = -1;
        // SAFETY: as in ScopedDeviceIndex::try_new_at.
        assert_eq!(
            unsafe { wwr_sys::wwr_get_device(&mut d) },
            wwr_status::WWR_STATUS_SUCCESS
        );
        d
    }

    /// Needs a GPU: device 0 must exist and have room for a context.
    #[test]
    fn switches_and_restores() {
        let before = current();
        {
            let guard = ScopedDeviceIndex::new(0);
            assert_eq!(guard.original_idx, before);
            assert_eq!(current(), 0);
        }
        assert_eq!(current(), before);
    }

    /// Needs a GPU. A bad index is an error at the caller's line, and the
    /// current device is left as it was.
    #[test]
    fn a_bad_index_is_an_error_at_the_caller() {
        let before = current();
        let line = line!() + 1;
        let error = ScopedDeviceIndex::try_new(-1).err().unwrap();
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert!(error.message.starts_with("WWR_STATUS_INVALID_DEVICE"));
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
        assert_eq!(current(), before);
    }
}
