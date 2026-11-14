//! wwr's `src/extension/blas` (module `wwr.extension.blas`): the BLAS
//! status as an error code, and the owning BLAS handle bound to a stream.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:blas_error`, the `wwrblasStatus_t` specializations | `impl ErrorCode for wwr_blas_status` |
//! | `:blas_handle`, `BlasHandleWrapper<P_create, P_destroy, S, P_device>` | [`BlasHandle`], over [`Stream`] |
//! | `concept blas_handle` | nothing: a call takes the raw handle, [`BlasHandle`]'s `get()` |
//!
//! As in wwr, the handle is a [`StreamBoundHandle`]: made from a shared
//! stream owner, created on its device, bound to its stream, and holding
//! the owner so the stream outlives it. Its constructor aborts on failure,
//! as every wrapper's in [`runtime`](crate::runtime) does.
//!
//! **No workspace yet.** cuBLAS and hipBLAS allocate a workspace per handle
//! unless handed one, and the tree's design hands each handle a worker row
//! of the arena (`docs/gpu_layer.md`, "Layering"); WarpWraps' C ABI has no
//! `wwr_blas_set_workspace` to hand it through, so the handle allocates its
//! own when it is made.

use std::ops::Deref;
use std::rc::Rc;

use wwr_sys::{wwr_blas_handle, wwr_blas_status, wwr_stream};

use crate::common::ErrorCode;
use crate::handle::{
    DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandle, StreamBoundKind,
};
use crate::runtime::Stream;
use fb_core_error_handling::check_or_abort;

/// `wwrblasStatus_t`, as the C ABI hands it back. The name is the
/// enumerator's; wwr's would be the vendor's (`CUBLAS_STATUS_ALLOC_FAILED`),
/// which the C ABI maps away.
impl ErrorCode for wwr_blas_status {
    const SUCCESS: Self = wwr_blas_status::WWR_BLAS_STATUS_SUCCESS;

    fn name(self) -> &'static str {
        use wwr_blas_status::*;
        match self {
            WWR_BLAS_STATUS_SUCCESS => "WWR_BLAS_STATUS_SUCCESS",
            WWR_BLAS_STATUS_NOT_INITIALIZED => "WWR_BLAS_STATUS_NOT_INITIALIZED",
            WWR_BLAS_STATUS_ALLOC_FAILED => "WWR_BLAS_STATUS_ALLOC_FAILED",
            WWR_BLAS_STATUS_ARCH_MISMATCH => "WWR_BLAS_STATUS_ARCH_MISMATCH",
            WWR_BLAS_STATUS_EXECUTION_FAILED => "WWR_BLAS_STATUS_EXECUTION_FAILED",
            WWR_BLAS_STATUS_INTERNAL_ERROR => "WWR_BLAS_STATUS_INTERNAL_ERROR",
            WWR_BLAS_STATUS_INVALID_VALUE => "WWR_BLAS_STATUS_INVALID_VALUE",
            WWR_BLAS_STATUS_MAPPING_ERROR => "WWR_BLAS_STATUS_MAPPING_ERROR",
            WWR_BLAS_STATUS_NOT_SUPPORTED => "WWR_BLAS_STATUS_NOT_SUPPORTED",
            WWR_BLAS_STATUS_UNKNOWN => "WWR_BLAS_STATUS_UNKNOWN",
        }
    }

    fn description(self) -> &'static str {
        use wwr_blas_status::*;
        match self {
            WWR_BLAS_STATUS_SUCCESS => "no error",
            WWR_BLAS_STATUS_NOT_INITIALIZED => "the library was not initialized",
            WWR_BLAS_STATUS_ALLOC_FAILED => "a resource allocation failed",
            WWR_BLAS_STATUS_ARCH_MISMATCH => "the device lacks a feature the call needs",
            WWR_BLAS_STATUS_EXECUTION_FAILED => "the GPU program failed to execute",
            WWR_BLAS_STATUS_INTERNAL_ERROR => "an internal library operation failed",
            WWR_BLAS_STATUS_INVALID_VALUE => "an unsupported value or parameter",
            WWR_BLAS_STATUS_MAPPING_ERROR => "access to GPU memory space failed",
            WWR_BLAS_STATUS_NOT_SUPPORTED => "the functionality is not supported",
            WWR_BLAS_STATUS_UNKNOWN => "a backend error with no stable code of its own",
        }
    }
}

/// A BLAS handle's create, destroy and stream bind.
pub struct BlasKind;

impl HandleKind for BlasKind {
    type Raw = wwr_blas_handle;
    type Error = wwr_blas_status;

    unsafe fn create(raw: *mut wwr_blas_handle) -> wwr_blas_status {
        // SAFETY: the caller hands a pointer valid for one write.
        unsafe { wwr_sys::wwr_blas_create(raw) }
    }

    unsafe fn destroy(raw: wwr_blas_handle) -> wwr_blas_status {
        // SAFETY: `raw` is a live handle, destroyed once.
        unsafe { wwr_sys::wwr_blas_destroy(raw) }
    }
}

impl StreamBoundKind for BlasKind {
    unsafe fn set_stream(raw: wwr_blas_handle, stream: wwr_stream) -> wwr_blas_status {
        // SAFETY: the caller's: both are live.
        unsafe { wwr_sys::wwr_blas_set_stream(raw, stream) }
    }
}

/// An owned BLAS handle bound to a stream it keeps alive:
/// `BlasHandleWrapper`. Every call through it enqueues on that stream.
///
/// Like the stream it is bound to, it is neither `Send` nor `Sync`; an
/// owner that shares it across threads makes sure no two call through it at
/// once.
pub struct BlasHandle(StreamBoundHandle<BlasKind, Stream>);

impl BlasHandle {
    /// A handle on `owner`'s device, bound to its stream. Aborts on
    /// failure, logging at critical against the caller.
    #[track_caller]
    pub fn new(owner: Rc<Stream>) -> Self {
        BlasHandle(check_or_abort(StreamBoundHandle::try_new(owner)))
    }
}

impl Deref for BlasHandle {
    type Target = StreamBoundHandle<BlasKind, Stream>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeviceHandle for BlasHandle {
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

impl DeviceHandleStream for BlasHandle {
    /// The bound stream.
    fn stream(&self) -> wwr_stream {
        self.0.stream()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_status;
    use crate::runtime::tests::ok;

    #[test]
    fn every_status_has_a_name_and_a_description() {
        use wwr_blas_status::*;
        for s in [
            WWR_BLAS_STATUS_SUCCESS,
            WWR_BLAS_STATUS_NOT_INITIALIZED,
            WWR_BLAS_STATUS_ALLOC_FAILED,
            WWR_BLAS_STATUS_ARCH_MISMATCH,
            WWR_BLAS_STATUS_EXECUTION_FAILED,
            WWR_BLAS_STATUS_INTERNAL_ERROR,
            WWR_BLAS_STATUS_INVALID_VALUE,
            WWR_BLAS_STATUS_MAPPING_ERROR,
            WWR_BLAS_STATUS_NOT_SUPPORTED,
            WWR_BLAS_STATUS_UNKNOWN,
        ] {
            assert_eq!(s.name(), format!("{s:?}"));
            assert!(!s.description().is_empty());
        }
        assert_eq!(
            gpu_status(WWR_BLAS_STATUS_INVALID_VALUE)
                .unwrap_err()
                .message,
            "WWR_BLAS_STATUS_INVALID_VALUE (an unsupported value or parameter)"
        );
    }

    /// Needs a GPU. The handle lands on the owner's device and the library
    /// reads back the owner's stream as the one it enqueues on.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn binds_the_owners_stream() {
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        assert_eq!(blas.dev_idx(), 0);
        assert_eq!(blas.stream(), owner.stream());
        let mut bound: wwr_stream = std::ptr::null_mut();
        // SAFETY: the handle is live; writes one stream through a local.
        let status = unsafe { wwr_sys::wwr_blas_get_stream(blas.get(), &mut bound) };
        assert_eq!(status, wwr_blas_status::WWR_BLAS_STATUS_SUCCESS);
        assert_eq!(bound, owner.stream());
        drop(blas);
        assert_eq!(Rc::strong_count(&owner), 1);
        ok(owner.sync());
    }
}
