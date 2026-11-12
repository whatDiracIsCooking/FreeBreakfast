//! `wwr.extension.handle:device_bound_handle_view`.

use std::ops::Deref;

use contracts::debug_requires;

use super::{DeviceHandle, HandleView};

/// A [`HandleView`] that also carries its handle's device:
/// `DeviceBoundHandleView<T>`. It never asks the runtime; the device is
/// whatever the owner recorded, or -1 when unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBoundHandleView<'a, T: Copy> {
    view: HandleView<'a, T>,
    dev_idx: i32,
}

impl<T: Copy> DeviceBoundHandleView<'_, T> {
    /// View `raw`, on device `dev_idx` (-1 if unknown), which the caller
    /// keeps alive for as long as it uses it.
    #[debug_requires(dev_idx >= -1, "a device index is -1 (unknown) or more")]
    pub fn from_raw(raw: T, dev_idx: i32) -> Self {
        DeviceBoundHandleView {
            view: HandleView::from_raw(raw),
            dev_idx,
        }
    }
}

impl<T: Copy> DeviceHandle for DeviceBoundHandleView<'_, T> {
    fn dev_idx(&self) -> i32 {
        self.dev_idx
    }
}

impl<'a, T: Copy> Deref for DeviceBoundHandleView<'a, T> {
    type Target = HandleView<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_an_unknown_device() {
        assert_eq!(DeviceBoundHandleView::from_raw(3u64, -1).dev_idx(), -1);
    }

    #[test]
    #[cfg(debug_assertions)] // the contract is compiled out of release
    #[should_panic(expected = "a device index is -1 (unknown) or more")]
    fn refuses_a_device_below_minus_one() {
        DeviceBoundHandleView::from_raw(3u64, -2);
    }
}
