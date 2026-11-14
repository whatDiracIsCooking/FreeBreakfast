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
//! **The workspace.** cuBLAS and hipBLAS allocate a workspace per handle
//! unless handed one, and the tree's design hands each handle a worker row
//! of the arena (`docs/gpu_layer.md`, "Layering"):
//! [`BlasHandle::set_workspace`] hands it through, [`WORKSPACE_BYTES`] long.
//!
use std::ffi::c_void;
use std::ops::Deref;
use std::rc::Rc;

use wwr_sys::{wwr_blas_handle, wwr_blas_status, wwr_stream};

use crate::common::ErrorCode;
use crate::gpu_status;
use crate::handle::{
    DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandle, StreamBoundKind,
};
use crate::runtime::Stream;
use fb_core_error_handling::{Status, check_or_abort};

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

/// The bytes of device memory each BLAS handle is handed as its workspace:
/// 32 MiB, cuBLAS's own default on sm_90, the largest it allocates for
/// itself on any architecture (4 MiB below sm_90). Too little makes cuBLAS
/// pick slower kernels rather than fail.
pub const WORKSPACE_BYTES: usize = 32 << 20;

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

    /// Hand the handle `bytes` of device memory at `workspace` as its
    /// workspace, in place of the one it allocates for itself:
    /// `wwrblasSetWorkspace`. Every later call through the handle uses it.
    ///
    /// # Errors
    /// The library's refusal, as a `GpuRuntime` error naming the status:
    /// cuBLAS refuses a pointer not 256-byte aligned.
    ///
    /// # Safety
    /// `workspace` is `bytes` of device memory on the handle's device,
    /// 256-byte aligned, and stays allocated, and written by nothing else,
    /// for as long as any call through the handle may still run.
    pub unsafe fn set_workspace(&self, workspace: *mut c_void, bytes: usize) -> Status {
        // SAFETY: the handle is live; the caller's for the memory.
        gpu_status(unsafe { wwr_sys::wwr_blas_set_workspace(self.get(), workspace, bytes) })
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
    use crate::runtime::tests::{DeviceBytes, ok};
    use wwr_sys::wwr_blas_operation;

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

    /// Needs a GPU. A handle handed a [`WORKSPACE_BYTES`] workspace runs a
    /// DGEMM through it to the exact product: small integers, so every
    /// partial sum is exact in any order.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn runs_a_gemm_on_a_workspace_it_was_handed() {
        const N: usize = 16;
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let workspace = DeviceBytes::new(WORKSPACE_BYTES);
        assert_eq!(workspace.0 as usize % 256, 0);
        // SAFETY: a live allocation of WORKSPACE_BYTES on device 0, which
        // outlives every call through the handle below.
        unsafe { blas.set_workspace(workspace.0, WORKSPACE_BYTES) }.unwrap();

        let a: Vec<f64> = (0..N * N).map(|i| (i % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..N * N).map(|i| (i % 5) as f64 - 2.0).collect();
        let mut want = vec![0.0; N * N];
        for j in 0..N {
            for i in 0..N {
                want[i + j * N] = (0..N).map(|k| a[i + k * N] * b[k + j * N]).sum();
            }
        }
        let (da, db) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
        );
        let dc = DeviceBytes::new(N * N * size_of::<f64>());
        let (one, zero) = (1.0f64, 0.0f64);
        let n = N as i32;
        // SAFETY: three live device buffers of N² doubles; the handle is
        // live and bound to `owner`'s stream.
        unsafe {
            gpu_status(wwr_sys::wwr_blas_dgemm(
                blas.get(),
                wwr_blas_operation::WWR_BLAS_OP_N,
                wwr_blas_operation::WWR_BLAS_OP_N,
                n,
                n,
                n,
                &one,
                da.0.cast(),
                n,
                db.0.cast(),
                n,
                &zero,
                dc.0.cast(),
                n,
            ))
            .unwrap();
        }
        let got: Vec<f64> = dc
            .read(&owner)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_ne_bytes(*c))
            .collect();
        assert_eq!(got, want);
        drop(blas);
        ok(owner.sync());
    }
}
