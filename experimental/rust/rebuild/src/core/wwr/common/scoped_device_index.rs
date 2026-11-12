//! `wwr.extension.common:scoped_device_index`.

use std::marker::PhantomData;

use super::{ErrorPolicy, gpu_check};
use wwr_sys::wwr_status;

/// Make a device current for the guard's lifetime, and restore the one that
/// was current before when it drops.
///
/// All three runtime calls, the get and set that enter and the set that
/// restores, go through the one policy `P`, retained for the guard's life and
/// readable through [`ScopedDeviceIndex::device_policy`].
///
/// As in wwr, a failed `get` under a policy that returns leaves
/// `original_idx` at -1, and the restore then reports a second failure.
///
/// The C++ guard cannot be moved. This one can, harmlessly: the current
/// device is a property of the thread, not of the guard's address. It is
/// neither `Send` nor `Sync`, though, because restoring on another thread
/// would set that thread's device and leave this one's switched.
pub struct ScopedDeviceIndex<P: ErrorPolicy<Error = wwr_status>> {
    /// The device current at construction, restored on drop.
    pub original_idx: i32,
    policy_device: P,
    not_send: PhantomData<*const ()>,
}

impl<P: ErrorPolicy<Error = wwr_status>> ScopedDeviceIndex<P> {
    /// Switch to `target_idx`, recording the current device to restore.
    #[track_caller]
    pub fn new(target_idx: i32, mut policy_device: P) -> Self {
        let mut original_idx = -1;
        // SAFETY: a plain runtime call writing one i32 through a valid local.
        gpu_check(
            unsafe { wwr_sys::wwr_get_device(&mut original_idx) },
            &mut policy_device,
        );
        // SAFETY: a plain runtime call; a bad index is a status, not UB.
        gpu_check(
            unsafe { wwr_sys::wwr_set_device(target_idx) },
            &mut policy_device,
        );
        ScopedDeviceIndex {
            original_idx,
            policy_device,
            not_send: PhantomData,
        }
    }

    /// The policy that handled this guard's device calls, for reading back
    /// what a stateful one recorded.
    pub fn device_policy(&self) -> &P {
        &self.policy_device
    }
}

impl<P: ErrorPolicy<Error = wwr_status>> Drop for ScopedDeviceIndex<P> {
    fn drop(&mut self) {
        // SAFETY: as in new. The location reported is this line, as wwr's
        // destructor reports its own.
        gpu_check(
            unsafe { wwr_sys::wwr_set_device(self.original_idx) },
            &mut self.policy_device,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Recording;
    use super::*;

    fn current() -> i32 {
        let mut d = -1;
        // SAFETY: as in ScopedDeviceIndex::new.
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
            let guard = ScopedDeviceIndex::new(0, Recording::default());
            assert_eq!(guard.original_idx, before);
            assert_eq!(current(), 0);
            assert_eq!(guard.device_policy().0, []);
        }
        assert_eq!(current(), before);
    }

    /// Needs a GPU. A bad index fails the set through the policy, at the
    /// caller's line; the get and the restore still succeed.
    #[test]
    fn reports_a_bad_index_at_the_caller() {
        let line = line!() + 1;
        let guard = ScopedDeviceIndex::new(-1, Recording::default());
        assert_eq!(
            guard.device_policy().0,
            [(wwr_status::WWR_STATUS_INVALID_DEVICE, line)]
        );
    }
}
