//! wwr's `src/extension/blas` (module `wwr.extension.blas`): the BLAS
//! status as an error code, and the owning BLAS handle bound to a stream.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:blas_error`, the `wwrblasStatus_t` specializations | `impl ErrorCode for wwr_blas_status` |
//! | `:blas_handle`, `BlasHandleWrapper<P_create, P_destroy, S, P_device>` | [`BlasHandle`], over [`Stream`] |
//! | `concept blas_handle` | nothing: a call takes the raw handle, [`BlasHandle`]'s `get()` |
//! | `usual_fp`, the typed dispatch `gemm<T>` over `wwrblas{S,D,C,Z}gemm` | [`BlasScalar`], its `gemm_raw` |
//! | the same `gemm<T>`, called with raw pointers | [`gemm`], over [`DeviceSlice`]s it checks |
//! | `dgmm<T>` over `wwrblas{S,D,C,Z}dgmm` | [`BlasScalar`]'s `dgmm_raw` |
//! | the same `dgmm<T>`, called with raw pointers | [`dgmm`], over [`DeviceSlice`]s it checks; `incx` at least 1, and C never A (no in-place call) |
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
//! **The typed calls.** [`BlasScalar`] is wwr's typed dispatch: one trait
//! method per routine, implemented for the four types the library computes
//! in, each forwarding to its letter's C ABI call. It takes raw pointers
//! and is `unsafe`, as the C ABI is; only this crate calls it. [`gemm`] and
//! [`dgmm`] are the safe calls over it: their operands are
//! [`DeviceSlice`]s, each extent
//! the call will touch is checked against the slice's length before
//! anything is enqueued, and a slice too short panics, as an index out of
//! a slice's bounds does. That check is what makes it safe, so it is on in
//! every build. The conveniences above it (packed operands, ops and sides
//! fixed at compile time) are `fb-recipes-blas`'s, which holds no `unsafe`.
//!
//! **What the safe calls leave out.** The library's `dgmm` may write C over
//! A, but a slice cannot be borrowed shared and exclusive at once, so
//! [`dgmm`] offers no in-place form: C is always memory of its own. Its
//! stride `incx` is at least 1: the library's 0 (one scale for every
//! row or column) and negative strides (x read backwards) are refused, as
//! nothing in the tree asks for them.

use std::ffi::c_void;
use std::ops::Deref;
use std::rc::Rc;

use wwr_sys::{
    wwr_blas_handle, wwr_blas_operation, wwr_blas_side_mode, wwr_blas_status, wwr_complex64,
    wwr_complex128, wwr_stream,
};

use crate::common::ErrorCode;
use crate::device_slice::{DeviceSlice, DeviceSliceMut};
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

/// A type the BLAS computes in: `f32`, `f64`, [`wwr_complex64`] and
/// [`wwr_complex128`], wwr's `usual_fp`. Each method is the C ABI call of
/// the type's letter. Sealed, since [`gemm`] and [`dgmm`] trust each method
/// to be that call.
pub trait BlasScalar: Copy + sealed::Sealed {
    /// `wwrblas<t>gemm`: `C = alpha op_a(A) op_b(B) + beta C`, column-major,
    /// `op_a(A)` m × k, `op_b(B)` k × n, C m × n.
    ///
    /// # Safety
    /// The library's: `handle` is live; A, B and C are device memory on its
    /// device covering their extents at `lda`, `ldb` and `ldc`; C apart
    /// from A and B; `alpha` and `beta` read as the handle's pointer mode
    /// says; all of it alive until the work enqueued on the handle's
    /// stream has finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn gemm_raw(
        handle: wwr_blas_handle,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> wwr_blas_status;

    /// `wwrblas<t>dgmm`: `C = A diag(x)` (`WWR_BLAS_SIDE_RIGHT`, x of n
    /// entries) or `C = diag(x) A` (`_LEFT`, m entries), column-major, A and
    /// C m × n, x read at stride `incx`.
    ///
    /// # Safety
    /// The library's: `handle` is live; A, x and C are device memory on its
    /// device covering their extents at `lda`, `incx` and `ldc`; C either A
    /// itself at the same leading dimension or apart from A and x; all of
    /// it alive until the work enqueued on the handle's stream has
    /// finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn dgmm_raw(
        handle: wwr_blas_handle,
        side: wwr_blas_side_mode,
        m: i32,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        c: *mut Self,
        ldc: i32,
    ) -> wwr_blas_status;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! blas_scalar {
    ($($t:ty => $gemm:ident, $dgmm:ident);* $(;)?) => {$(
        impl sealed::Sealed for $t {}

        impl BlasScalar for $t {
            unsafe fn gemm_raw(
                handle: wwr_blas_handle,
                op_a: wwr_blas_operation,
                op_b: wwr_blas_operation,
                m: i32,
                n: i32,
                k: i32,
                alpha: *const Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: *const Self,
                c: *mut Self,
                ldc: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe {
                    wwr_sys::$gemm(
                        handle, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }

            unsafe fn dgmm_raw(
                handle: wwr_blas_handle,
                side: wwr_blas_side_mode,
                m: i32,
                n: i32,
                a: *const Self,
                lda: i32,
                x: *const Self,
                incx: i32,
                c: *mut Self,
                ldc: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$dgmm(handle, side, m, n, a, lda, x, incx, c, ldc) }
            }
        }
    )*};
}

blas_scalar! {
    f32 => wwr_blas_sgemm, wwr_blas_sdgmm;
    f64 => wwr_blas_dgemm, wwr_blas_ddgmm;
    wwr_complex64 => wwr_blas_cgemm, wwr_blas_cdgmm;
    wwr_complex128 => wwr_blas_zgemm, wwr_blas_zdgmm;
}

/// Panic unless a column-major `rows` × `cols` matrix at leading dimension
/// `ld` fits in `len` elements: `ld` is at least `max(1, rows)`, as the
/// library requires, and the last column ends inside the slice. `routine`
/// names the call in the message.
#[track_caller]
pub(crate) fn check_extent(routine: &str, name: &str, len: usize, rows: i32, cols: i32, ld: i32) {
    assert!(
        ld >= rows.max(1),
        "{routine}: {name}'s leading dimension {ld} is less than max(1, {rows} rows)"
    );
    if rows == 0 || cols == 0 {
        return;
    }
    // In u64, where a product of two i32s cannot overflow.
    let need = ld as u64 * (cols as u64 - 1) + rows as u64;
    assert!(
        need <= len as u64,
        "{routine}: {name} is {rows} x {cols} at leading dimension {ld}, {need} \
         elements, but its slice holds {len}"
    );
}

/// Panic unless a vector of `count` entries at stride `inc` fits in `len`
/// elements: `inc` is at least 1 (the safe calls take no zero or negative
/// stride), and the last entry, at `(count - 1) inc`, is inside the slice.
#[track_caller]
pub(crate) fn check_vector(routine: &str, name: &str, len: usize, count: i32, inc: i32) {
    assert!(inc >= 1, "{routine}: {name}'s stride {inc} is less than 1");
    if count == 0 {
        return;
    }
    // In u64, where a product of two i32s cannot overflow.
    let need = 1 + (count as u64 - 1) * inc as u64;
    assert!(
        need <= len as u64,
        "{routine}: {name} is {count} entries at stride {inc}, {need} elements, \
         but its slice holds {len}"
    );
}

/// `C = alpha op_a(A) op_b(B) + beta C` on `handle`'s stream: column-major,
/// `op_a(A)` m × k, `op_b(B)` k × n, C m × n, each at its leading dimension.
/// wwr's `gemm<T>`, with every matrix a slice it checks.
///
/// A is stored m × k under `WWR_BLAS_OP_N` and k × m under `_T` or `_C`
/// (the conjugate transpose, the same as `_T` on a real `T`), B likewise
/// k × n or n × k. `alpha` and `beta` are host values, read in the handle's
/// default pointer mode. With `beta` zero, C need not be initialised. The
/// borrow checker keeps C apart from A and B.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If m, n or k is negative, a leading dimension is less than its matrix's
/// row count (or 1), or a slice is shorter than its matrix at its leading
/// dimension; before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[track_caller]
pub fn gemm<T: BlasScalar>(
    handle: &BlasHandle,
    op_a: wwr_blas_operation,
    op_b: wwr_blas_operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    b: &impl DeviceSlice<T>,
    ldb: i32,
    beta: T,
    c: &mut impl DeviceSliceMut<T>,
    ldc: i32,
) -> Status {
    assert!(
        m >= 0 && n >= 0 && k >= 0,
        "gemm: m, n and k are {m}, {n} and {k}"
    );
    let stored = |op, rows, cols| {
        if op == wwr_blas_operation::WWR_BLAS_OP_N {
            (rows, cols)
        } else {
            (cols, rows)
        }
    };
    let (a_rows, a_cols) = stored(op_a, m, k);
    let (b_rows, b_cols) = stored(op_b, k, n);
    check_extent("gemm", "A", a.len(), a_rows, a_cols, lda);
    check_extent("gemm", "B", b.len(), b_rows, b_cols, ldb);
    check_extent("gemm", "C", c.len(), m, n, ldc);
    let c = c.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); C is its memory's only user
    // (DeviceSliceMut's), so apart from A and B; the scalars are host
    // values, the handle's default pointer mode.
    gpu_status(unsafe {
        T::gemm_raw(
            handle.get(),
            op_a,
            op_b,
            m,
            n,
            k,
            &alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            &beta,
            c,
            ldc,
        )
    })
}

/// `C = A diag(x)` (`side` `WWR_BLAS_SIDE_RIGHT`) or `C = diag(x) A`
/// (`_LEFT`) on `handle`'s stream: column-major, A and C m × n at their
/// leading dimensions, x the diagonal, n entries on the right and m on the
/// left, at stride `incx`. wwr's `dgmm<T>`, with every operand a slice it
/// checks: the scale of each column (right) or row (left) of A by its
/// entry of x, in O(mn) where a gemm against the dense diagonal would be
/// O(mn²) or O(m²n).
///
/// The library also writes C over A; here C is a slice of its own, which
/// the borrow checker keeps apart from A and x, so there is no in-place
/// call. `incx` is at least 1: the library's 0 and negative strides are
/// refused.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If m or n is negative, a leading dimension is less than m (or 1),
/// `incx` is less than 1, or a slice is shorter than its operand's extent
/// (x's is `1 + (count - 1) incx` for a non-empty one); before anything is
/// enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[track_caller]
pub fn dgmm<T: BlasScalar>(
    handle: &BlasHandle,
    side: wwr_blas_side_mode,
    m: i32,
    n: i32,
    a: &impl DeviceSlice<T>,
    lda: i32,
    x: &impl DeviceSlice<T>,
    incx: i32,
    c: &mut impl DeviceSliceMut<T>,
    ldc: i32,
) -> Status {
    assert!(m >= 0 && n >= 0, "dgmm: m and n are {m} and {n}");
    let count = match side {
        wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT => m,
        wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT => n,
    };
    check_extent("dgmm", "A", a.len(), m, n, lda);
    check_vector("dgmm", "x", x.len(), count, incx);
    check_extent("dgmm", "C", c.len(), m, n, ldc);
    let c = c.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); C is its memory's only user
    // (DeviceSliceMut's), so apart from A and x.
    gpu_status(unsafe {
        T::dgmm_raw(
            handle.get(),
            side,
            m,
            n,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            c,
            ldc,
        )
    })
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
        let n = N as i32;
        let want = host_gemm(
            false,
            false,
            (N, N, N),
            1.0,
            (&a, N),
            (&b, N),
            0.0,
            (&[], N),
        );
        let (da, db) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
        );
        let mut dc = DeviceBytes::new(N * N * size_of::<f64>());
        gemm(
            &blas, OP_N, OP_N, n, n, n, 1.0, &da, n, &db, n, 0.0, &mut dc, n,
        )
        .unwrap();
        assert_eq!(download(&dc, &owner), want);
        drop(blas);
        ok(owner.sync());
    }

    use wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

    /// An `ld` × `cols` column-major array, padding included, of small
    /// integers, so every product and partial sum is exact in any order.
    fn matrix(ld: usize, cols: usize, seed: usize) -> Vec<f64> {
        (0..ld * cols)
            .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
            .collect()
    }

    /// `alpha op_a(A) op_b(B) + beta C` on the host, column-major, `(m, n,
    /// k)`, each matrix with its leading dimension; C's padding (rows past
    /// m) kept. An empty C reads as zero.
    #[allow(clippy::too_many_arguments)]
    fn host_gemm(
        t_a: bool,
        t_b: bool,
        (m, n, k): (usize, usize, usize),
        alpha: f64,
        (a, lda): (&[f64], usize),
        (b, ldb): (&[f64], usize),
        beta: f64,
        (c, ldc): (&[f64], usize),
    ) -> Vec<f64> {
        let mut out = if c.is_empty() {
            vec![0.0; ldc * n]
        } else {
            c.to_vec()
        };
        for j in 0..n {
            for i in 0..m {
                let dot: f64 = (0..k)
                    .map(|l| {
                        let x = if t_a { a[l + i * lda] } else { a[i + l * lda] };
                        let y = if t_b { b[j + l * ldb] } else { b[l + j * ldb] };
                        x * y
                    })
                    .sum();
                out[i + j * ldc] = alpha * dot + beta * out[i + j * ldc];
            }
        }
        out
    }

    fn download(device: &DeviceBytes, owner: &Stream) -> Vec<f64> {
        device
            .read(owner)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_ne_bytes(*c))
            .collect()
    }

    /// Needs a GPU. Transposed operands, a rectangular shape, every leading
    /// dimension padded and a non-zero beta: the product is exact, and C's
    /// padding rows are left as they were.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemm_reads_each_operand_at_its_op_and_leading_dimension() {
        let (m, n, k) = (5, 3, 4);
        let (lda, ldb, ldc) = (k + 2, n + 1, m + 3);
        // A is stored k × m (op T), B n × k (op T), C m × n.
        let (a, b, c) = (matrix(lda, m, 1), matrix(ldb, k, 2), matrix(ldc, n, 3));
        let want = host_gemm(
            true,
            true,
            (m, n, k),
            -2.0,
            (&a, lda),
            (&b, ldb),
            3.0,
            (&c, ldc),
        );
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, db, mut dc) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
            DeviceBytes::upload(&c, &owner),
        );
        let i = |x: usize| x as i32;
        gemm(
            &blas,
            OP_T,
            OP_T,
            i(m),
            i(n),
            i(k),
            -2.0,
            &da,
            i(lda),
            &db,
            i(ldb),
            3.0,
            &mut dc,
            i(ldc),
        )
        .unwrap();
        assert_eq!(download(&dc, &owner), want);
    }

    /// Needs a GPU. A C one element short of its extent panics before
    /// anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "gemm: C is 4 x 4 at leading dimension 4, 16 elements, but its slice holds 15"
    )]
    fn gemm_refuses_a_short_slice() {
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(16 * size_of::<f64>());
        let mut c = DeviceBytes::new(15 * size_of::<f64>());
        let _ = gemm(
            &blas, OP_N, OP_N, 4, 4, 4, 1.0, &a, 4, &a, 4, 0.0, &mut c, 4,
        );
    }

    #[test]
    fn an_extent_fits_to_its_last_element() {
        // 3 × 2 at ld 5: the second column ends at element 5 + 3 = 8.
        check_extent("gemm", "A", 8, 3, 2, 5);
        // Nothing to touch, but the library still wants ld >= 1.
        check_extent("gemm", "A", 0, 0, 7, 1);
        check_extent("gemm", "A", 0, 3, 0, 3);
    }

    #[test]
    #[should_panic(
        expected = "gemm: B is 3 x 2 at leading dimension 5, 8 elements, but its slice holds 7"
    )]
    fn an_extent_one_past_the_slice_panics() {
        check_extent("gemm", "B", 7, 3, 2, 5);
    }

    #[test]
    #[should_panic(expected = "gemm: C's leading dimension 2 is less than max(1, 3 rows)")]
    fn a_leading_dimension_under_the_rows_panics() {
        check_extent("gemm", "C", 100, 3, 2, 2);
    }

    #[test]
    #[should_panic(expected = "dgmm: A's leading dimension 0 is less than max(1, 0 rows)")]
    fn a_zero_leading_dimension_panics() {
        check_extent("dgmm", "A", 0, 0, 0, 0);
    }

    #[test]
    fn a_vector_fits_to_its_last_entry() {
        // 4 entries at stride 3: the last at element 9, so 10 in all.
        check_vector("dgmm", "x", 10, 4, 3);
        check_vector("dgmm", "x", 4, 4, 1);
        // Nothing to touch, but the stride is still checked.
        check_vector("dgmm", "x", 0, 0, 1);
    }

    #[test]
    #[should_panic(
        expected = "dgmm: x is 4 entries at stride 3, 10 elements, but its slice holds 9"
    )]
    fn a_vector_one_past_the_slice_panics() {
        check_vector("dgmm", "x", 9, 4, 3);
    }

    #[test]
    #[should_panic(expected = "dgmm: x's stride 0 is less than 1")]
    fn a_zero_stride_panics() {
        check_vector("dgmm", "x", 100, 0, 0);
    }

    #[test]
    #[should_panic(expected = "dgmm: x's stride -1 is less than 1")]
    fn a_negative_stride_panics() {
        check_vector("dgmm", "x", 100, 4, -1);
    }

    use wwr_blas_side_mode::{WWR_BLAS_SIDE_LEFT as LEFT, WWR_BLAS_SIDE_RIGHT as RIGHT};

    /// `diag(x) A` (`left`) or `A diag(x)` on the host, column-major, m × n,
    /// A at `lda`, x at `incx`, into C at `ldc` with C's padding kept.
    #[allow(clippy::too_many_arguments)]
    fn host_dgmm(
        left: bool,
        (m, n): (usize, usize),
        (a, lda): (&[f64], usize),
        (x, incx): (&[f64], usize),
        (c, ldc): (&[f64], usize),
    ) -> Vec<f64> {
        let mut out = c.to_vec();
        for j in 0..n {
            for i in 0..m {
                let d = x[if left { i } else { j } * incx];
                out[i + j * ldc] = d * a[i + j * lda];
            }
        }
        out
    }

    /// One dgmm on a card with padded leading dimensions and a strided x,
    /// checked bitwise against the host's; C's padding rows left as they
    /// were.
    fn check_dgmm(left: bool) {
        let (m, n) = (5, 3);
        let (lda, ldc, incx) = (m + 2, m + 3, 2);
        let count = if left { m } else { n };
        let (a, x, c) = (
            matrix(lda, n, 1),
            matrix(1 + (count - 1) * incx, 1, 2),
            matrix(ldc, n, 3),
        );
        let want = host_dgmm(left, (m, n), (&a, lda), (&x, incx), (&c, ldc));
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, dx, mut dc) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&x, &owner),
            DeviceBytes::upload(&c, &owner),
        );
        let i = |v: usize| v as i32;
        let side = if left { LEFT } else { RIGHT };
        dgmm::<f64>(
            &blas,
            side,
            i(m),
            i(n),
            &da,
            i(lda),
            &dx,
            i(incx),
            &mut dc,
            i(ldc),
        )
        .unwrap();
        assert_eq!(download(&dc, &owner), want);
    }

    /// Needs a GPU. `A diag(x)`: each column of A scaled by its entry of x,
    /// every leading dimension padded and x strided.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn dgmm_scales_each_column_on_the_right() {
        check_dgmm(false);
    }

    /// Needs a GPU. `diag(x) A`: each row scaled, x m entries long.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn dgmm_scales_each_row_on_the_left() {
        check_dgmm(true);
    }

    /// Needs a GPU. On the left x has m entries, so an x of n (fewer) is
    /// short, and panics before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "dgmm: x is 4 entries at stride 1, 4 elements, but its slice holds 3"
    )]
    fn dgmm_refuses_a_short_slice() {
        let owner = Rc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(12 * size_of::<f64>());
        let x = DeviceBytes::new(3 * size_of::<f64>());
        let mut c = DeviceBytes::new(12 * size_of::<f64>());
        let _ = dgmm::<f64>(&blas, LEFT, 4, 3, &a, 4, &x, 1, &mut c, 4);
    }
}
