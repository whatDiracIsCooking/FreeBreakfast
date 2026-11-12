//! wwr's `src/extension/handle` (module `wwr.extension.handle`): the RAII
//! GPU-handle layer, owning handles and the views that borrow them.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:handle`, `BaseHandle<T, Derived, P_create, P_destroy>` | [`BaseHandle<K, PC, PD>`](BaseHandle), with `Derived`'s `create`/`destroy` hooks as the [`HandleKind`] `K` (`base_handle.rs`) |
//! | `:device_bound_handle`, `DeviceBoundHandle` | [`DeviceBoundHandle`], which derefs to its `BaseHandle` |
//! | `:stream_bound_handle`, `StreamBoundHandle` | [`StreamBoundHandle`], its `set_stream` hook as [`StreamBoundKind`] |
//! | `:handle_view`, `:device_bound_handle_view`, `:stream_bound_handle_view` | [`HandleView`], [`DeviceBoundHandleView`], [`StreamBoundHandleView`], each borrowing the handle it came from |
//! | `:device_handle`, the `device_handle` concept ladder | [`DeviceHandle`], [`DeviceHandleStream`], [`DeviceHandlePool`] |
//!
//! CRTP's `static_cast<Derived *>(this)->create(...)` is a call to the kind's
//! associated function, and public inheritance is a field plus `Deref`, so
//! an outer type's `view()` hides its base's as in C++. The policy slots wwr
//! collapses with `policy_slot` are plain fields: a zero-sized policy costs
//! nothing however many slots share its type.

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
pub use device_handle::{DeviceHandle, DeviceHandlePool, DeviceHandleStream};
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
    use crate::core::wwr::common::kit::AbortPolicy;

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
            unsafe { wwr_sys::wwr_stream_create(raw) }
        }
        unsafe fn destroy(raw: wwr_stream) -> wwr_status {
            unsafe { wwr_sys::wwr_stream_destroy(raw) }
        }
    }

    pub(crate) struct BoundEventKind;

    impl HandleKind for BoundEventKind {
        type Raw = wwr_event;
        type Error = wwr_status;
        unsafe fn create(raw: *mut wwr_event) -> wwr_status {
            unsafe { wwr_sys::wwr_event_create(raw) }
        }
        unsafe fn destroy(raw: wwr_event) -> wwr_status {
            unsafe { wwr_sys::wwr_event_destroy(raw) }
        }
    }

    impl StreamBoundKind for BoundEventKind {
        unsafe fn set_stream(raw: wwr_event, stream: wwr_stream) -> wwr_status {
            unsafe { wwr_sys::wwr_event_record(raw, stream) }
        }
    }

    type Abort = AbortPolicy<wwr_status>;

    /// A stream owner, as the runtime module's `Stream` will be.
    pub(crate) struct OwnedStream(DeviceBoundHandle<StreamKind, Abort, Abort, Abort>);

    impl OwnedStream {
        pub(crate) fn new(dev_idx: i32) -> Self {
            OwnedStream(DeviceBoundHandle::new(
                dev_idx,
                Abort::new(),
                Abort::new(),
                Abort::new(),
            ))
        }
    }

    impl DeviceHandle for OwnedStream {
        fn dev_idx(&self) -> i32 {
            self.0.dev_idx()
        }
    }

    impl DeviceHandleStream for OwnedStream {
        fn stream(&self) -> wwr_stream {
            self.0.get().unwrap()
        }
    }

    pub(crate) fn current_device() -> i32 {
        let mut d = -1;
        // SAFETY: writes one i32 through a valid local.
        assert_eq!(
            unsafe { wwr_sys::wwr_get_device(&mut d) },
            wwr_status::WWR_STATUS_SUCCESS
        );
        d
    }

    #[test]
    fn stateless_policies_cost_nothing() {
        // Three same-type empty policies: what wwr's policy_slot collapses.
        // The handle is its raw handle and its device index, and nothing more.
        assert_eq!(
            std::mem::size_of::<DeviceBoundHandle<StreamKind, Abort, Abort, Abort>>(),
            std::mem::size_of::<(Option<wwr_stream>, i32)>()
        );
    }
}
