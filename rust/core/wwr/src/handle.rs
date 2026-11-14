//! wwr's `src/extension/handle` (module `wwr.extension.handle`): the RAII
//! GPU-handle layer, owning handles and the views that borrow them.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:handle`, `BaseHandle<T, Derived, P_create, P_destroy>` | [`BaseHandle<K>`](BaseHandle), with `Derived`'s `create`/`destroy` hooks as the [`HandleKind`] `K` (`base_handle.rs`) |
//! | `:device_bound_handle`, `DeviceBoundHandle` | [`DeviceBoundHandle`], which derefs to its `BaseHandle` |
//! | `:stream_bound_handle`, `StreamBoundHandle` | [`StreamBoundHandle`], its `set_stream` hook as [`StreamBoundKind`] |
//! | `:handle_view`, `:device_bound_handle_view`, `:stream_bound_handle_view` | [`HandleView`], [`DeviceBoundHandleView`], [`StreamBoundHandleView`], each borrowing the handle it came from |
//! | `:device_handle`, the `device_handle` concept ladder | [`DeviceHandle`], [`DeviceHandleStream`]; `device_handle_pool` has none (`device_handle.rs`) |
//!
//! CRTP's `static_cast<Derived *>(this)->create(...)` is a call to the kind's
//! associated function, and public inheritance is a field plus `Deref`, so
//! an outer type's `view()` hides its base's as in C++.
//!
//! **No policy parameters.** wwr's `P_create`, `P_destroy` and `P_device`
//! decide what a failed GPU call in a constructor or destructor does. Here a
//! constructor is `try_new`, which returns `Result`, and a `Drop` logs a
//! failed destroy at error; the wrappers above decide to abort on an `Err`
//! ([`crate::common`] has the reasons).

mod base_handle;
mod device_bound_handle;
mod device_bound_handle_view;
mod device_handle;
mod handle_view;
mod stream_bound_handle;
mod stream_bound_handle_view;

pub use base_handle::{BaseHandle, HandleKind};
pub use device_bound_handle::DeviceBoundHandle;
pub use device_bound_handle_view::DeviceBoundHandleView;
pub use device_handle::{DeviceHandle, DeviceHandleStream};
pub use handle_view::HandleView;
pub use stream_bound_handle::{StreamBoundHandle, StreamBoundKind};
pub use stream_bound_handle_view::StreamBoundHandleView;

/// Kinds the tests build handles of: a host-only fake, a real stream, and an
/// event that "binds" a stream by being recorded on it.
#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use wwr_sys::{wwr_event, wwr_status, wwr_stream};

    use super::*;

    thread_local! {
        static FAKE_DESTROYED: Cell<u32> = const { Cell::new(0) };
        static FAKE_FAIL_NEXT: Cell<bool> = const { Cell::new(false) };
    }

    /// Handles `1, 2, ...` with no GPU behind them; counts its destroys, per
    /// thread, so parallel tests do not see each other's.
    pub(crate) struct Fake;

    pub(crate) fn fake_destroyed() -> u32 {
        FAKE_DESTROYED.get()
    }

    /// Make the next `Fake` create on this thread fail.
    pub(crate) fn fake_fail_next_create() {
        FAKE_FAIL_NEXT.set(true);
    }

    impl HandleKind for Fake {
        type Raw = u64;
        type Error = wwr_status;
        unsafe fn create(raw: *mut u64) -> wwr_status {
            if FAKE_FAIL_NEXT.replace(false) {
                return wwr_status::WWR_STATUS_MEMORY_ALLOCATION;
            }
            // SAFETY: the caller hands a pointer valid for one write.
            unsafe { *raw = 1 };
            wwr_status::WWR_STATUS_SUCCESS
        }
        unsafe fn destroy(_: u64) -> wwr_status {
            FAKE_DESTROYED.set(FAKE_DESTROYED.get() + 1);
            wwr_status::WWR_STATUS_SUCCESS
        }
    }

    pub(crate) struct StreamKind;

    impl HandleKind for StreamKind {
        type Raw = wwr_stream;
        type Error = wwr_status;
        unsafe fn create(raw: *mut wwr_stream) -> wwr_status {
            // SAFETY: the caller hands a pointer valid for one write.
            unsafe { wwr_sys::wwr_stream_create(raw) }
        }
        unsafe fn destroy(raw: wwr_stream) -> wwr_status {
            // SAFETY: the caller hands a live stream, used never again.
            unsafe { wwr_sys::wwr_stream_destroy(raw) }
        }
    }

    pub(crate) struct BoundEventKind;

    impl HandleKind for BoundEventKind {
        type Raw = wwr_event;
        type Error = wwr_status;
        unsafe fn create(raw: *mut wwr_event) -> wwr_status {
            // SAFETY: the caller hands a pointer valid for one write.
            unsafe { wwr_sys::wwr_event_create(raw) }
        }
        unsafe fn destroy(raw: wwr_event) -> wwr_status {
            // SAFETY: the caller hands a live event, used never again.
            unsafe { wwr_sys::wwr_event_destroy(raw) }
        }
    }

    impl StreamBoundKind for BoundEventKind {
        unsafe fn set_stream(raw: wwr_event, stream: wwr_stream) -> wwr_status {
            // SAFETY: the caller hands a live event and a live stream.
            unsafe { wwr_sys::wwr_event_record(raw, stream) }
        }
    }

    /// A stream owner standing in for `runtime::Stream`: `handle` sits below
    /// `runtime`, so its tests do not reach up to it, as wwr's cannot.
    pub(crate) struct OwnedStream(DeviceBoundHandle<StreamKind>);

    impl OwnedStream {
        #[track_caller]
        pub(crate) fn new(dev_idx: i32) -> Self {
            OwnedStream(DeviceBoundHandle::try_new(dev_idx).unwrap())
        }
    }

    impl DeviceHandle for OwnedStream {
        fn dev_idx(&self) -> i32 {
            self.0.dev_idx()
        }
    }

    impl DeviceHandleStream for OwnedStream {
        fn stream(&self) -> wwr_stream {
            self.0.get()
        }
    }

    pub(crate) fn current_device() -> i32 {
        let mut d = -1;
        // SAFETY: writes one i32 through a valid local.
        let status = unsafe { wwr_sys::wwr_get_device(&mut d) };
        assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
        d
    }

    #[test]
    fn a_handle_is_its_raw_handle_and_device_and_nothing_more() {
        // No liveness flag, no policy fields.
        assert_eq!(
            std::mem::size_of::<BaseHandle<StreamKind>>(),
            std::mem::size_of::<wwr_stream>()
        );
        assert_eq!(
            std::mem::size_of::<DeviceBoundHandle<StreamKind>>(),
            std::mem::size_of::<(wwr_stream, i32)>()
        );
    }
}
