//! wwr's `src/extension/runtime` (module `wwr.extension.runtime`): owning
//! wrappers for the runtime API's streams, events, memory pools and graphs,
//! built on [`handle`](super::handle).
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:stream`, `StreamWrapper<P_create, P_destroy, P_device>` | [`Stream<PC, PD, PDev>`](Stream) |
//! | `:event`, `EventWrapper` | [`Event`] |
//! | `:mem_pool`, `MemPoolWrapper`, and the `wwrMemPoolProps` it fills | [`MemPool`], [`MemPoolProps`] |
//! | `:graph`, `GraphWrapper` | [`Graph`] |
//! | `:graph_exec`, `GraphExecWrapper` | [`GraphExec`] |
//! | `:device_handle`, `kit::DeviceHandle` | [`kit::DeviceHandle`] |
//! | `StreamView`, `EventView`, `GraphExecView` | the handle views [`view()`](super::handle::DeviceBoundHandle::view) returns |
//!
//! Each wrapper is a newtype over the handle it is built on, and derefs to
//! it, so `get()`, `view()`, `dev_idx()` and the policies are the handle's.
//! Its `HandleKind` is crate-private: the wrapper is the only way to make
//! one.
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
//! call them safely. A method on an owner that holds nothing returns
//! `WWR_STATUS_INVALID_VALUE` without calling the runtime, except a
//! stream's, which falls back to the legacy default stream as wwr's
//! `stream()` does. wwr overloads `sync` on `wwrStream_t` against
//! `wwrEvent_t`; the C ABI erases both to `*mut c_void`, so the raw forms
//! are told apart by module path, and the type system cannot.
//!
//! **Flags pass through untranslated.** The C ABI forwards a flags word to
//! the backend as it is, so the constants here ([`stream::NON_BLOCKING`],
//! [`event::DISABLE_TIMING`], [`event::BLOCKING_SYNC`]) are the vendors'
//! values, which CUDA and HIP share.
//!
//! **Left out:** `kit::DeviceHandle::props()`. `wwrDeviceProp` reaches the
//! C ABI as a bare `void*` whose layout differs between the backends, so
//! there is nothing safe to fill.

pub mod event;
pub mod graph;
pub mod graph_exec;
pub mod kit;
pub mod mem_pool;
pub mod stream;

pub use event::Event;
pub use graph::Graph;
pub use graph_exec::GraphExec;
pub use mem_pool::{MemPool, MemPoolProps};
pub use stream::Stream;

use wwr_sys::wwr_status;

use super::common::{ErrorPolicy, SourceLocation, gpu_check_at};

/// Run a runtime create that writes a raw handle, reporting through
/// `policy`, and yield the handle if it is live: what each wrapper's
/// non-default constructor hands [`new_with`](super::handle::BaseHandle::new_with).
fn create_with<T, P: ErrorPolicy<Error = wwr_status>>(
    policy: &mut P,
    location: SourceLocation,
    create: impl FnOnce(*mut T) -> wwr_status,
) -> Option<T> {
    let mut raw = std::mem::MaybeUninit::uninit();
    let ok = gpu_check_at(create(raw.as_mut_ptr()), policy, location);
    // SAFETY: a runtime create writes its handle when it succeeds.
    ok.then(|| unsafe { raw.assume_init() })
}

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

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::c_void;

    use wwr_sys::{wwr_memcpy_kind, wwr_status, wwr_stream};

    use crate::core::wwr::common::kit::AbortPolicy;

    pub(crate) type Abort = AbortPolicy<wwr_status>;

    #[track_caller]
    pub(crate) fn ok(status: wwr_status) {
        assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
    }

    /// `n` bytes of device memory, freed on drop.
    pub(crate) struct DeviceBytes(pub *mut c_void, usize);

    impl DeviceBytes {
        pub(crate) fn new(n: usize) -> Self {
            let mut p = std::ptr::null_mut();
            // SAFETY: writes one pointer through a valid local.
            ok(unsafe { wwr_sys::wwr_malloc(&mut p, n) });
            DeviceBytes(p, n)
        }

        /// The bytes, after the work queued on `stream` has finished.
        pub(crate) fn read(&self, stream: wwr_stream) -> Vec<u8> {
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
