//! wwr's `src/extension/runtime` (module `wwr.extension.runtime`): owning
//! wrappers for the runtime API's streams, events, memory pools and graphs,
//! built on [`handle`](super::handle).
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:stream`, `StreamWrapper<P_create, P_destroy, P_device>` | [`Stream`] |
//! | `:event`, `EventWrapper` | [`Event`] |
//! | `:mem_pool`, `MemPoolWrapper`, and the `wwrMemPoolProps` it fills | [`MemPool`], [`MemPoolProps`] |
//! | `:graph`, `GraphWrapper` | [`Graph`] |
//! | `:graph_exec`, `GraphExecWrapper` | [`GraphExec`] |
//! | `:device_handle`, `kit::DeviceHandle` | nothing: see below |
//! | `StreamView`, `EventView`, `GraphExecView` | the handle views [`view()`](super::handle::DeviceBoundHandle::view) returns |
//!
//! Each wrapper is a newtype over the handle it is built on, and derefs to
//! it, so `get()`, `view()` and `dev_idx()` are the handle's. Its
//! `HandleKind` is crate-private: the wrapper is the only way to make one.
//!
//! **A constructor aborts on failure**, logging at critical against its
//! caller: the tree's rule for every create (`docs/diagnostics.md`,
//! "Allocation is a planning-time error"), which the C++ tree spells
//! `CreatePolicy`. Each wrapper builds on one private fallible constructor
//! that returns the `Err` instead; a site that must recover from a failed
//! create is the reason to make that one public, and nothing else changes.
//! A wrapper is therefore always live, and no method has a case for one
//! that holds nothing.
//!
//! **The borrow-safe operations are methods on the owners**, and `unsafe`
//! free functions on the raw handle beside them. wwr's free functions take
//! a raw handle so one definition serves an owner, a view and a bare
//! handle; a destroyed handle is a use-after-free inside the runtime,
//! which only an owner's borrow rules out, so here the raw forms
//! ([`stream::sync`], [`stream::wait_event`], [`stream::begin_capture`],
//! [`event::record`], [`event::record_with_flags`], [`event::sync`],
//! [`graph_exec::launch`], [`graph_exec::upload`]) are `unsafe`, and
//! [`Stream::sync`], [`Event::record`], [`GraphExec::launch`] and the rest
//! call them safely. wwr overloads `sync` on `wwrStream_t` against
//! `wwrEvent_t`; the C ABI erases both to `*mut c_void`, so the raw forms
//! are told apart by module path, and the type system cannot.
//!
//! **Flags pass through untranslated.** The C ABI forwards a flags word to
//! the backend as it is, so the constants here ([`stream::NON_BLOCKING`],
//! [`event::DISABLE_TIMING`], [`event::BLOCKING_SYNC`]) are the vendors'
//! values, which CUDA and HIP share.
//!
//! **`kit::DeviceHandle`**, an owned stream and memory pool on one device,
//! is wwr's ready-made top rung of the device-handle ladder, for a buffer
//! that allocates from a pool. Neither the rung nor the bundle has a
//! counterpart: the arena allocates one plain block per plan
//! (`docs/memory.md`, "The two kinds"), and a [`Stream`] is a
//! [`DeviceHandleStream`](super::handle::DeviceHandleStream) on its own.

pub mod event;
pub mod graph;
pub mod graph_exec;
pub mod mem_pool;
pub mod stream;

pub use event::Event;
pub use graph::Graph;
pub use graph_exec::GraphExec;
pub use mem_pool::{MemPool, MemPoolProps};
pub use stream::Stream;

use wwr_sys::wwr_status;

/// The first failure of a destroy that is two runtime calls, both made:
/// wwr synchronizes a stream or event before destroying it and reports
/// each. A `HandleKind` returns one status, so the sync's failure wins,
/// and the destroy still runs.
fn first_failure(sync: wwr_status, destroy: wwr_status) -> wwr_status {
    if sync == wwr_status::WWR_STATUS_SUCCESS {
        destroy
    } else {
        sync
    }
}

/// Device helpers for this crate's tests and, through the `test-support`
/// feature, for a dependent's.
#[cfg(any(test, feature = "test-support"))]
pub mod tests {
    use std::ffi::c_void;

    use wwr_sys::{wwr_memcpy_kind, wwr_status, wwr_stream};

    use fb_core_error_handling::{Error, ErrorKind};

    /// Assert that `error` is a GPU failure whose status is named `name`,
    /// reported at `line` of `file`.
    #[track_caller]
    pub fn assert_gpu_error_at(error: Error, name: &str, file: &str, line: u32) {
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert!(error.message.starts_with(name), "{}", error.message);
        assert_eq!((error.origin.file(), error.origin.line()), (file, line));
    }

    #[track_caller]
    pub fn ok(status: wwr_status) {
        assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
    }

    /// `n` bytes of device memory, freed on drop.
    pub struct DeviceBytes(pub *mut c_void, usize);

    impl DeviceBytes {
        pub fn new(n: usize) -> Self {
            let mut p = std::ptr::null_mut();
            // SAFETY: writes one pointer through a valid local.
            ok(unsafe { wwr_sys::wwr_malloc(&mut p, n) });
            DeviceBytes(p, n)
        }

        /// The bytes, after the work queued on `stream` has finished.
        #[allow(
            clippy::not_unsafe_ptr_arg_deref,
            reason = "a test helper; `pub` only for the test-support feature, \
                      and every caller passes a live stream's handle"
        )]
        pub fn read(&self, stream: wwr_stream) -> Vec<u8> {
            let mut host = vec![0u8; self.1];
            // SAFETY: both ranges are `self.1` bytes; the stream is drained
            // before the synchronous copy reads the device side.
            unsafe {
                ok(wwr_sys::wwr_stream_synchronize(stream));
                ok(wwr_sys::wwr_memcpy(
                    host.as_mut_ptr().cast(),
                    self.0,
                    self.1,
                    wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                ));
            }
            host
        }
    }

    impl Drop for DeviceBytes {
        fn drop(&mut self) {
            // SAFETY: allocated by `wwr_malloc`, freed once.
            ok(unsafe { wwr_sys::wwr_free(self.0) });
        }
    }

    // SAFETY: a live `wwr_malloc` allocation of `self.1` bytes on the
    // device current when it was made, 256-byte aligned, so as many whole
    // `T`s as fit; freed only on drop. A test using it on a handle of that
    // device, and writing it only through one `&mut`, keeps the rest.
    unsafe impl<T> crate::device_slice::DeviceSlice<T> for DeviceBytes {
        fn as_ptr(&self) -> *const T {
            self.0.cast_const().cast()
        }

        fn len(&self) -> usize {
            self.1 / size_of::<T>()
        }
    }

    // SAFETY: as above; `&mut self` makes the caller its only user.
    unsafe impl<T> crate::device_slice::DeviceSliceMut<T> for DeviceBytes {
        fn as_mut_ptr(&mut self) -> *mut T {
            self.0.cast()
        }
    }

    #[cfg(test)]
    #[test]
    fn first_failure_prefers_the_sync() {
        use super::first_failure;
        use wwr_status::*;
        assert_eq!(
            first_failure(WWR_STATUS_SUCCESS, WWR_STATUS_SUCCESS),
            WWR_STATUS_SUCCESS
        );
        assert_eq!(
            first_failure(WWR_STATUS_SUCCESS, WWR_STATUS_UNKNOWN),
            WWR_STATUS_UNKNOWN
        );
        assert_eq!(
            first_failure(WWR_STATUS_NOT_READY, WWR_STATUS_UNKNOWN),
            WWR_STATUS_NOT_READY
        );
    }
}
