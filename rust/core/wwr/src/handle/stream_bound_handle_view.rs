//! `wwr.extension.handle:stream_bound_handle_view`.

use std::ops::Deref;

use contracts::debug_requires;
use wwr_sys::wwr_stream;

use super::{DeviceBoundHandleView, DeviceHandle, DeviceHandleStream};

/// A [`DeviceBoundHandleView`] that also carries the work stream its handle
/// is bound to: `StreamBoundHandleView<T>`. The stream is a bare handle,
/// captured when the view was taken; the view retains nothing, and its
/// lifetime ties it to the owning handle, which does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamBoundHandleView<'a, T: Copy> {
    view: DeviceBoundHandleView<'a, T>,
    stream: wwr_stream,
}

impl<T: Copy> StreamBoundHandleView<'_, T> {
    /// View `raw`, on device `dev_idx` (-1 if unknown) and bound to `stream`
    /// (null if unknown), which the caller keeps alive for as long as it uses
    /// them.
    #[debug_requires(dev_idx >= -1, "a device index is -1 (unknown) or more")]
    pub fn from_raw(raw: T, dev_idx: i32, stream: wwr_stream) -> Self {
        StreamBoundHandleView {
            view: DeviceBoundHandleView::from_raw(raw, dev_idx),
            stream,
        }
    }
}

impl<T: Copy> DeviceHandle for StreamBoundHandleView<'_, T> {
    fn dev_idx(&self) -> i32 {
        self.view.dev_idx()
    }
}

impl<T: Copy> DeviceHandleStream for StreamBoundHandleView<'_, T> {
    fn stream(&self) -> wwr_stream {
        self.stream
    }
}

impl<'a, T: Copy> Deref for StreamBoundHandleView<'a, T> {
    type Target = DeviceBoundHandleView<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

#[cfg(all(test, debug_assertions))] // the contract is compiled out of release
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "a device index is -1 (unknown) or more")]
    fn refuses_a_device_below_minus_one() {
        StreamBoundHandleView::from_raw(3u64, -2, std::ptr::null_mut());
    }
}
