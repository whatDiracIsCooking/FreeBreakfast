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
//! | `gemmStridedBatched<T>` over `wwrblas{S,D,C,Z}gemmStridedBatched` | [`BlasScalar`]'s `gemm_strided_batched_raw` |
//! | the same `gemmStridedBatched<T>`, called with raw pointers | [`gemm_strided_batched`], over [`DeviceSlice`]s it checks at every stride; C's matrices apart from one another |
//! | `gemv<T>` over `wwrblas{S,D,C,Z}gemv` | [`BlasScalar`]'s `gemv_raw` |
//! | the same `gemv<T>`, called with raw pointers | [`gemv`], over [`DeviceSlice`]s it checks; `incx` and `incy` at least 1 |
//! | `syrk<T>` over `wwrblas{S,D,C,Z}syrk` | [`BlasScalar`]'s `syrk_raw` |
//! | the same `syrk<T>`, called with raw pointers | [`syrk`], over [`DeviceSlice`]s it checks; C never A (no in-place call) |
//! | `trsv<T>` and `scal<T>` over `wwrblas{S,D,C,Z}trsv` and `…scal` | [`BlasScalar`]'s `trsv_raw` and `scal_raw` |
//! | the same `trsv<T>` and `scal<T>`, called with raw pointers | [`trsv`] and [`scal`], over [`DeviceSlice`]s they check; the stride at least 1 |
//! | `gemm<T>` with B and C blocks of rows of one matrix, as `src/recipes/trsm`'s update calls it | [`gemm_rows`], the two blocks checked apart |
//! | `trsm<T>` over `wwrblas{S,D,C,Z}trsm` | not bound: the library's trsm takes workspace that grows with the right-hand sides on ROCm, so the tree's own is `fb_recipes_launch::trsm`, over [`gemm`], [`gemm_rows`], [`trsv`] and [`scal`] |
//! | `geam<T>` over `wwrblas{S,D,C,Z}geam` | [`BlasScalar`]'s `geam_raw` |
//! | the same `geam<T>`, called with raw pointers | [`geam`], over [`DeviceSlice`]s it checks; C never A or B (no in-place call) |
//! | `kOne<T>`, `kZero<T>` (`core/constants`), the BLAS scalars one and zero by address | [`BlasScalar::ONE`] and [`BlasScalar::ZERO`], by value |
//!
//! As in wwr, the handle is a [`StreamBoundHandle`]: made from a shared
//! stream owner, created on its device, bound to its stream, and holding
//! the owner so the stream outlives it. Its constructor aborts on failure,
//! as every wrapper's in [`runtime`](crate::runtime) does.
//!
//! **The workspace.** cuBLAS and hipBLAS allocate a workspace per handle
//! unless handed one, and the tree's design hands each handle a worker row
//! of the arena (`docs/gpu_layer.md`, "Layering"):
//! [`BlasHandle::bind_workspace`] hands it through. The library keeps the
//! pointer for every later call, beyond any borrow, so the handle takes the
//! memory by value, a [`DeviceSliceMut`] it owns until it is destroyed or
//! handed another, and no borrow-scoped form exists: resetting a handle to
//! no workspace at all is vendor-dependent.
//!
//! **The typed calls.** [`BlasScalar`] is wwr's typed dispatch: one trait
//! method per routine, implemented for the four types the library computes
//! in, each forwarding to its letter's C ABI call. It takes raw pointers
//! and is `unsafe`, as the C ABI is; only this crate calls it. [`gemm`],
//! [`gemm_rows`], [`gemm_strided_batched`], [`gemv`], [`dgmm`], [`syrk`],
//! [`geam`], [`trsv`] and [`scal`] are the safe calls over it: their operands are [`DeviceSlice`]s, each extent
//! the call will touch is checked against the slice's length before
//! anything is enqueued, and a slice too short panics, as an index out of
//! a slice's bounds does. That check is what makes it safe, so it is on in
//! every build. The conveniences above it (packed operands, ops and sides
//! fixed at compile time) are `fb-recipes-blas`'s, which holds no `unsafe`.
//!
//! **What the safe calls leave out.** The library's `dgmm` may write C over
//! A, but a slice cannot be borrowed shared and exclusive at once, so
//! [`dgmm`] offers no in-place form: C is always memory of its own. Its
//! stride `incx` is at least 1, as [`gemv`]'s are: the library's 0 (one
//! entry for every row or column) and negative strides (a vector read
//! backwards) are refused, as nothing in the tree asks for them. Likewise [`syrk`]'s
//! C is memory of its own, apart from A: a caller whose operands are
//! blocks of one matrix cuts it into disjoint parts first
//! ([`split_at_mut`](crate::device_slice::split_at_mut)), and where the
//! blocks are rows of the same columns, which no cut parts,
//! [`gemm_rows`] takes the matrix whole and checks the rows apart. So is
//! [`geam`]'s C, apart from A and B: the library's in-place forms (C over A
//! or over B) are left out with `dgmm`'s.

use std::ops::Deref;
use std::panic::Location;
use std::sync::Arc;

use wwr_sys::{
    wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_handle, wwr_blas_operation,
    wwr_blas_side_mode, wwr_blas_status, wwr_complex64, wwr_complex128, wwr_stream,
};

use crate::common::ErrorCode;
use crate::device_slice::{DeviceSlice, DeviceSliceMut};
use crate::handle::{
    DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandle, StreamBoundKind,
};
use crate::runtime::Stream;
use crate::{gpu_log_error_at, gpu_status};
use contracts::requires;
use fb_core_error_handling::{Status, check_or_abort};
use fb_core_log::{fits_i32, le};

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
/// `Send`, not `Sync`: a step that calls the library moves to whichever
/// worker picks it up, and its handles with it, but a handle's state (its
/// stream, its workspace) is one caller's at a time, so it is never shared.
///
/// ```compile_fail,E0277
/// fn shared_across_threads<T: Sync>() {}
/// shared_across_threads::<fb_core_wwr::blas::BlasHandle>();
/// ```
pub struct BlasHandle {
    // Fields drop in declaration order: the handle is destroyed before the
    // workspace it was handed is released.
    handle: StreamBoundHandle<BlasKind, Stream>,
    /// The memory handed to the library as its workspace, owned while the
    /// library holds its pointer; `None` while the handle is on the one it
    /// allocated for itself.
    workspace: Option<Box<dyn DeviceSliceMut<u8> + Send + Sync>>,
}

// SAFETY: the raw handle is a library object, not tied to the thread that
// made it: cuBLAS and hipBLAS allow a handle to be used from any thread,
// one call at a time, and every call names it. Moving the owner moves the
// pointer, the `Arc` of its stream (`Stream` is `Send` and `Sync`) and the
// workspace (`Send` by its bound); the handle is destroyed once, by its
// drop, on whichever thread that runs. It is not `Sync`, so no two threads
// call through one handle at once.
unsafe impl Send for BlasHandle {}

// The one above, checked at every build.
const _: () = {
    const fn moved_between_threads<T: Send>() {}
    moved_between_threads::<BlasHandle>();
};

impl BlasHandle {
    /// A handle on `owner`'s device, bound to its stream. Aborts on
    /// failure, logging at critical against the caller.
    #[track_caller]
    pub fn new(owner: Arc<Stream>) -> Self {
        BlasHandle {
            handle: check_or_abort(StreamBoundHandle::try_new(owner)),
            workspace: None,
        }
    }

    /// Hand the handle `workspace` as its workspace, in place of the one it
    /// allocated for itself or was handed before: `wwrblasSetWorkspace`.
    /// Every later call through the handle uses it, and the handle owns it
    /// until it is destroyed or handed another, since the library keeps the
    /// pointer. A previous workspace is released only after the library has
    /// the new pointer and the stream has finished every call enqueued
    /// before, so nothing still running uses memory that is gone.
    ///
    /// `workspace` must be on the handle's device, as every
    /// [`DeviceSlice`] a call through the handle takes must be; `'static`
    /// because the handle keeps it, so it owns its memory rather than
    /// borrowing it.
    ///
    /// # Errors
    /// The library's refusal, as a `GpuRuntime` error naming the status
    /// (cuBLAS refuses a pointer not 256-byte aligned), and the handle keeps
    /// its previous workspace; or a failed sync of the stream before a
    /// previous one is released, which keeps that one alive.
    #[allow(
        clippy::disallowed_methods,
        reason = "a rebind drains the stream before releasing the old workspace; it takes `&mut self`, so it runs at instantiation, outside any run"
    )]
    pub fn bind_workspace<W>(&mut self, workspace: W) -> Status
    where
        W: DeviceSliceMut<u8> + Send + Sync + 'static,
    {
        let mut workspace: Box<dyn DeviceSliceMut<u8> + Send + Sync> = Box::new(workspace);
        let (ptr, bytes) = (workspace.as_mut_ptr(), workspace.len());
        // SAFETY: the handle is live. `workspace` is `bytes` of device
        // memory on the handle's device that nothing else reads or writes
        // while it lives (`DeviceSliceMut`), and the handle owns it from
        // here until after the library has stopped using the pointer: it
        // is released only by the next bind, once the library holds that
        // one's pointer and the stream has drained, or after the handle is
        // destroyed (field order, and the sync in `Drop`). `&mut self`
        // rules out a call through the handle on another thread meanwhile.
        gpu_status(unsafe {
            wwr_sys::wwr_blas_set_workspace(self.handle.get(), ptr.cast(), bytes)
        })?;
        if let Some(previous) = self.workspace.replace(workspace)
            && let Err(e) = gpu_status(self.handle.owner().sync())
        {
            // A call still queued may use it: leak it rather than release
            // memory the device may yet write.
            std::mem::forget(previous);
            return Err(e);
        }
        Ok(())
    }

    /// The workspace the handle was last handed, if any.
    pub fn workspace(&self) -> Option<&(dyn DeviceSliceMut<u8> + Send + Sync)> {
        self.workspace.as_deref()
    }
}

impl Drop for BlasHandle {
    /// With a workspace handed to it: sync the stream first, so no call
    /// through the handle is still running when the fields drop, the handle
    /// and then its workspace. A failed sync is logged, as a failed destroy
    /// is, and the workspace leaked rather than released under a call that
    /// may still be running.
    #[allow(
        clippy::disallowed_methods,
        reason = "a drop drains the stream before its workspace is released, as a destroy does: not a task"
    )]
    fn drop(&mut self) {
        if self.workspace.is_some() {
            let status = self.handle.owner().sync();
            if gpu_status(status).is_err() {
                gpu_log_error_at(status, Location::caller());
                std::mem::forget(self.workspace.take());
            }
        }
    }
}

impl Deref for BlasHandle {
    type Target = StreamBoundHandle<BlasKind, Stream>;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl DeviceHandle for BlasHandle {
    fn dev_idx(&self) -> i32 {
        self.handle.dev_idx()
    }
}

impl DeviceHandleStream for BlasHandle {
    /// The bound stream.
    fn stream(&self) -> wwr_stream {
        self.handle.stream()
    }
}

/// A type the BLAS computes in: `f32`, `f64`, [`wwr_complex64`] and
/// [`wwr_complex128`], wwr's `usual_fp`. Each method is the C ABI call of
/// the type's letter. Sealed, since [`gemm`], [`gemm_rows`],
/// [`gemm_strided_batched`], [`gemv`], [`dgmm`], [`syrk`], [`geam`],
/// [`trsv`] and [`scal`] trust each method to
/// be that call.
pub trait BlasScalar: Copy + sealed::Sealed {
    /// One (`1 + 0i` on a complex type), the `alpha` of a plain product:
    /// the C++ `kOne<T>`.
    const ONE: Self;

    /// Zero, the `beta` that overwrites C rather than accumulating into it:
    /// the C++ `kZero<T>`.
    const ZERO: Self;

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

    /// `wwrblas<t>gemmStridedBatched`: `batch` products
    /// `C_i = alpha op_a(A_i) op_b(B_i) + beta C_i`, each as
    /// [`BlasScalar::gemm_raw`]'s, operand `i` of each matrix `i` strides
    /// past its first.
    ///
    /// # Safety
    /// As [`BlasScalar::gemm_raw`], for every operand of every batch at its
    /// stride; no two of the C's overlap.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn gemm_strided_batched_raw(
        handle: wwr_blas_handle,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        stride_a: i64,
        b: *const Self,
        ldb: i32,
        stride_b: i64,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
        stride_c: i64,
        batch: i32,
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

    /// `wwrblas<t>trsv`: `op(A) x = b`, x written over b, column-major, A
    /// triangular n × n, its `fill` triangle read, x of n entries at its
    /// stride.
    ///
    /// # Safety
    /// The library's: `handle` is live; A and x are device memory on its
    /// device covering their extents at `lda` and `incx`; x apart from A;
    /// all of it alive until the work enqueued on the handle's stream has
    /// finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn trsv_raw(
        handle: wwr_blas_handle,
        fill: wwr_blas_fill_mode,
        op: wwr_blas_operation,
        diag: wwr_blas_diag_type,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    ) -> wwr_blas_status;

    /// `wwrblas<t>scal`: `x = alpha x`, x of n entries at its stride.
    ///
    /// # Safety
    /// The library's: `handle` is live; x is device memory on its device
    /// covering its extent at `incx`; `alpha` read as the handle's pointer
    /// mode says; all of it alive until the work enqueued on the handle's
    /// stream has finished.
    unsafe fn scal_raw(
        handle: wwr_blas_handle,
        n: i32,
        alpha: *const Self,
        x: *mut Self,
        incx: i32,
    ) -> wwr_blas_status;

    /// `wwrblas<t>syrk`: `C = alpha op(A) op(A)ᵀ + beta C` on C's `fill`
    /// triangle, column-major, `op(A)` n × k, C n × n.
    ///
    /// # Safety
    /// The library's: `handle` is live; A and C are device memory on its
    /// device covering their extents at `lda` and `ldc`; C apart from A;
    /// `alpha` and `beta` read as the handle's pointer mode says; all of
    /// it alive until the work enqueued on the handle's stream has
    /// finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn syrk_raw(
        handle: wwr_blas_handle,
        fill: wwr_blas_fill_mode,
        op: wwr_blas_operation,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        beta: *const Self,
        c: *mut Self,
        ldc: i32,
    ) -> wwr_blas_status;

    /// `wwrblas<t>gemv`: `y = alpha op(A) x + beta y`, column-major, A
    /// stored m × n, x of n entries and y of m under `WWR_BLAS_OP_N`, the
    /// other way round under `_T`, each vector read at its stride.
    ///
    /// # Safety
    /// The library's: `handle` is live; A, x and y are device memory on its
    /// device covering their extents at `lda`, `incx` and `incy`; y apart
    /// from A and x; `alpha` and `beta` read as the handle's pointer mode
    /// says; all of it alive until the work enqueued on the handle's
    /// stream has finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn gemv_raw(
        handle: wwr_blas_handle,
        op: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: *const Self,
        y: *mut Self,
        incy: i32,
    ) -> wwr_blas_status;

    /// `wwrblas<t>geam`: `C = alpha op_a(A) + beta op_b(B)`, column-major,
    /// C m × n, A stored m × n under `WWR_BLAS_OP_N` and n × m under `_T`
    /// or `_C`, B likewise.
    ///
    /// # Safety
    /// The library's: `handle` is live; A, B and C are device memory on its
    /// device covering their extents at `lda`, `ldb` and `ldc`; C either
    /// apart from A and B, or A (or B) itself at the same leading dimension
    /// under `WWR_BLAS_OP_N`; `alpha` and `beta` read as the handle's
    /// pointer mode says; all of it alive until the work enqueued on the
    /// handle's stream has finished.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    unsafe fn geam_raw(
        handle: wwr_blas_handle,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: *const Self,
        lda: i32,
        beta: *const Self,
        b: *const Self,
        ldb: i32,
        c: *mut Self,
        ldc: i32,
    ) -> wwr_blas_status;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! blas_scalar {
    ($($t:ty => $gemm:ident, $batched:ident, $dgmm:ident, $trsv:ident, $scal:ident, $syrk:ident,
        $gemv:ident, $geam:ident, $one:expr, $zero:expr);* $(;)?) => {$(
        impl sealed::Sealed for $t {}

        impl BlasScalar for $t {
            const ONE: Self = $one;
            const ZERO: Self = $zero;

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

            unsafe fn gemm_strided_batched_raw(
                handle: wwr_blas_handle,
                op_a: wwr_blas_operation,
                op_b: wwr_blas_operation,
                m: i32,
                n: i32,
                k: i32,
                alpha: *const Self,
                a: *const Self,
                lda: i32,
                stride_a: i64,
                b: *const Self,
                ldb: i32,
                stride_b: i64,
                beta: *const Self,
                c: *mut Self,
                ldc: i32,
                stride_c: i64,
                batch: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe {
                    wwr_sys::$batched(
                        handle, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                        beta, c, ldc, stride_c, batch,
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

            unsafe fn trsv_raw(
                handle: wwr_blas_handle,
                fill: wwr_blas_fill_mode,
                op: wwr_blas_operation,
                diag: wwr_blas_diag_type,
                n: i32,
                a: *const Self,
                lda: i32,
                x: *mut Self,
                incx: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$trsv(handle, fill, op, diag, n, a, lda, x, incx) }
            }

            unsafe fn scal_raw(
                handle: wwr_blas_handle,
                n: i32,
                alpha: *const Self,
                x: *mut Self,
                incx: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$scal(handle, n, alpha, x, incx) }
            }

            unsafe fn syrk_raw(
                handle: wwr_blas_handle,
                fill: wwr_blas_fill_mode,
                op: wwr_blas_operation,
                n: i32,
                k: i32,
                alpha: *const Self,
                a: *const Self,
                lda: i32,
                beta: *const Self,
                c: *mut Self,
                ldc: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$syrk(handle, fill, op, n, k, alpha, a, lda, beta, c, ldc) }
            }

            unsafe fn gemv_raw(
                handle: wwr_blas_handle,
                op: wwr_blas_operation,
                m: i32,
                n: i32,
                alpha: *const Self,
                a: *const Self,
                lda: i32,
                x: *const Self,
                incx: i32,
                beta: *const Self,
                y: *mut Self,
                incy: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$gemv(handle, op, m, n, alpha, a, lda, x, incx, beta, y, incy) }
            }

            unsafe fn geam_raw(
                handle: wwr_blas_handle,
                op_a: wwr_blas_operation,
                op_b: wwr_blas_operation,
                m: i32,
                n: i32,
                alpha: *const Self,
                a: *const Self,
                lda: i32,
                beta: *const Self,
                b: *const Self,
                ldb: i32,
                c: *mut Self,
                ldc: i32,
            ) -> wwr_blas_status {
                // SAFETY: the caller's, which are the call's.
                unsafe {
                    wwr_sys::$geam(handle, op_a, op_b, m, n, alpha, a, lda, beta, b, ldb, c, ldc)
                }
            }
        }
    )*};
}

blas_scalar! {
    f32 => wwr_blas_sgemm, wwr_blas_sgemm_strided_batched, wwr_blas_sdgmm, wwr_blas_strsv, wwr_blas_sscal,
        wwr_blas_ssyrk, wwr_blas_sgemv, wwr_blas_sgeam, 1.0, 0.0;
    f64 => wwr_blas_dgemm, wwr_blas_dgemm_strided_batched, wwr_blas_ddgmm, wwr_blas_dtrsv, wwr_blas_dscal,
        wwr_blas_dsyrk, wwr_blas_dgemv, wwr_blas_dgeam, 1.0, 0.0;
    wwr_complex64 => wwr_blas_cgemm, wwr_blas_cgemm_strided_batched, wwr_blas_cdgmm,
        wwr_blas_ctrsv, wwr_blas_cscal, wwr_blas_csyrk, wwr_blas_cgemv, wwr_blas_cgeam,
        wwr_complex64 { re: 1.0, im: 0.0 }, wwr_complex64 { re: 0.0, im: 0.0 };
    wwr_complex128 => wwr_blas_zgemm, wwr_blas_zgemm_strided_batched, wwr_blas_zdgmm,
        wwr_blas_ztrsv, wwr_blas_zscal, wwr_blas_zsyrk, wwr_blas_zgemv, wwr_blas_zgeam,
        wwr_complex128 { re: 1.0, im: 0.0 }, wwr_complex128 { re: 0.0, im: 0.0 };
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
/// If a leading dimension is less than its matrix's row count (or 1), or a
/// slice is shorter than its matrix at its leading dimension; before
/// anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n) && le!(0, k), "m, n and k are not negative")]
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

/// Panic unless `batch` column-major `rows` × `cols` matrices at leading
/// dimension `ld`, the `i`-th `i · stride` elements past the first, fit in
/// `len` elements: `stride` is non-negative, and, when `apart`, at least
/// one matrix's extent, so no two of them overlap (an operand the call
/// writes). An empty batch or matrix touches nothing.
#[track_caller]
#[allow(
    clippy::too_many_arguments,
    reason = "one matrix's extent and its batch"
)]
pub(crate) fn check_batch(
    routine: &str,
    name: &str,
    len: usize,
    rows: i32,
    cols: i32,
    ld: i32,
    stride: i64,
    batch: i32,
    apart: bool,
) {
    assert!(
        stride >= 0,
        "{routine}: {name}'s stride {stride} is negative"
    );
    // An empty batch reads nothing; its leading dimension is still checked.
    check_extent(
        routine,
        name,
        len,
        rows,
        if batch == 0 { 0 } else { cols },
        ld,
    );
    if batch == 0 || rows == 0 || cols == 0 {
        return;
    }
    // In u64, where a product of two i32s cannot overflow.
    let one = ld as u64 * (cols as u64 - 1) + rows as u64;
    assert!(
        !apart || batch == 1 || stride as u64 >= one,
        "{routine}: {name}'s stride {stride} is less than one matrix's {one} elements, so two \
         of the batch overlap"
    );
    let need = (stride as u64)
        .checked_mul(batch as u64 - 1)
        .and_then(|last| last.checked_add(one));
    assert!(
        need.is_some_and(|need| need <= len as u64),
        "{routine}: {name} is {batch} matrices {rows} x {cols} at leading dimension {ld} and \
         stride {stride}, but its slice holds {len} elements"
    );
}

/// `batch` products `C_i = alpha op_a(A_i) op_b(B_i) + beta C_i` on
/// `handle`'s stream, each as [`gemm`]'s, operand `i` of A, B and C
/// `i · stride` elements past its first. wwr's `gemmStridedBatched<T>`,
/// with every operand a slice it checks.
///
/// A stride of 0 for A or B shares one matrix with every product (the
/// transform of each block by one U); C's stride is at least one C's
/// extent, so no two products write the same element. With `beta` zero, C
/// need not be initialised. The borrow checker keeps C apart from A and B.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If a leading dimension is less than its matrix's row count (or 1), a
/// stride is negative, C's stride is less than one C's extent (with more
/// than one product), or a slice is shorter than its last matrix at its
/// stride; before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(
    le!(0, m) && le!(0, n) && le!(0, k) && le!(0, batch),
    "m, n, k and batch are not negative"
)]
#[track_caller]
pub fn gemm_strided_batched<T: BlasScalar>(
    handle: &BlasHandle,
    op_a: wwr_blas_operation,
    op_b: wwr_blas_operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    stride_a: i64,
    b: &impl DeviceSlice<T>,
    ldb: i32,
    stride_b: i64,
    beta: T,
    c: &mut impl DeviceSliceMut<T>,
    ldc: i32,
    stride_c: i64,
    batch: i32,
) -> Status {
    const ROUTINE: &str = "gemm_strided_batched";
    let stored = |op, rows, cols| {
        if op == wwr_blas_operation::WWR_BLAS_OP_N {
            (rows, cols)
        } else {
            (cols, rows)
        }
    };
    let (a_rows, a_cols) = stored(op_a, m, k);
    let (b_rows, b_cols) = stored(op_b, k, n);
    check_batch(
        ROUTINE,
        "A",
        a.len(),
        a_rows,
        a_cols,
        lda,
        stride_a,
        batch,
        false,
    );
    check_batch(
        ROUTINE,
        "B",
        b.len(),
        b_rows,
        b_cols,
        ldb,
        stride_b,
        batch,
        false,
    );
    check_batch(ROUTINE, "C", c.len(), m, n, ldc, stride_c, batch, true);
    let c = c.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering every
    // matrix of the batch at its stride, checked above, and stays so until
    // the enqueued work finishes (DeviceSlice's promise); C is its memory's
    // only user (DeviceSliceMut's), so apart from A and B, and its matrices
    // are apart from one another (checked above); the scalars are host
    // values, the handle's default pointer mode.
    gpu_status(unsafe {
        T::gemm_strided_batched_raw(
            handle.get(),
            op_a,
            op_b,
            m,
            n,
            k,
            &alpha,
            a.as_ptr(),
            lda,
            stride_a,
            b.as_ptr(),
            ldb,
            stride_b,
            &beta,
            c,
            ldc,
            stride_c,
            batch,
        )
    })
}

/// `y = alpha op(A) x + beta y` on `handle`'s stream: column-major, A
/// stored m × n at its leading dimension, x of n entries and y of m under
/// `WWR_BLAS_OP_N`, x of m and y of n under `_T` or `_C` (the same as `_T`
/// on a real `T`), each vector at its stride. wwr's `gemv<T>`, with every
/// operand a slice it checks.
///
/// `alpha` and `beta` are host values, read in the handle's default pointer
/// mode. With `beta` zero, y need not be initialised. The borrow checker
/// keeps y apart from A and x. The strides are at least 1: the library's 0
/// and negative ones are refused.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If A's leading dimension is less than m (or 1), a stride is less than
/// 1, or a slice is shorter than its operand's extent; before anything is
/// enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n), "m and n are not negative")]
#[track_caller]
pub fn gemv<T: BlasScalar>(
    handle: &BlasHandle,
    op: wwr_blas_operation,
    m: i32,
    n: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    x: &impl DeviceSlice<T>,
    incx: i32,
    beta: T,
    y: &mut impl DeviceSliceMut<T>,
    incy: i32,
) -> Status {
    let (x_count, y_count) = if op == wwr_blas_operation::WWR_BLAS_OP_N {
        (n, m)
    } else {
        (m, n)
    };
    check_extent("gemv", "A", a.len(), m, n, lda);
    check_vector("gemv", "x", x.len(), x_count, incx);
    check_vector("gemv", "y", y.len(), y_count, incy);
    let y = y.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); y is its memory's only user
    // (DeviceSliceMut's), so apart from A and x; the scalars are host
    // values, the handle's default pointer mode.
    gpu_status(unsafe {
        T::gemv_raw(
            handle.get(),
            op,
            m,
            n,
            &alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            &beta,
            y,
            incy,
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
/// If a leading dimension is less than m (or 1), `incx` is less than 1, or a slice is shorter than its operand's extent
/// (x's is `1 + (count - 1) incx` for a non-empty one); before anything is
/// enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n), "m and n are not negative")]
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

/// `B(dst.., :) = alpha op_a(A) B(src.., :) + beta B(dst.., :)` on
/// `handle`'s stream: a [`gemm`] whose B and C are two blocks of rows of
/// one column-major matrix `b` of n columns at leading dimension `ldb`,
/// the k rows from `src` read and the m rows from `dst` written, `op_a(A)`
/// m × k. The update between the diagonal blocks of a left-sided blocked
/// triangular solve (`fb_recipes_launch::trsm`), where the rows solved and the rows
/// still to come lie in the same columns, so no cut of `b` parts them
/// ([`split_at_mut`](crate::device_slice::split_at_mut)); here the two
/// blocks are checked apart instead.
///
/// A is stored m × k under `WWR_BLAS_OP_N` and k × m under `_T` or `_C`.
/// `alpha` and `beta` are host values, read in the handle's default pointer
/// mode. The rows of `b` outside the two blocks are neither read nor
/// written. The borrow checker keeps `b` apart from A.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If a leading dimension is less than its matrix's row count (or 1; for
/// `b`, the end of the later block), or a slice is shorter than its matrix
/// at its leading dimension; before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n) && le!(0, k), "m, n and k are not negative")]
#[requires(le!(0, src) && le!(0, dst), "the blocks' first rows are not negative")]
#[requires(
    fits_i32!(src.checked_add(k)) && fits_i32!(dst.checked_add(m)),
    "each block's end fits an i32"
)]
#[requires(
    src + k <= dst || dst + m <= src,
    "the rows read and the rows written are apart"
)]
#[track_caller]
pub fn gemm_rows<T: BlasScalar>(
    handle: &BlasHandle,
    op_a: wwr_blas_operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    b: &mut impl DeviceSliceMut<T>,
    ldb: i32,
    src: i32,
    beta: T,
    dst: i32,
) -> Status {
    let (a_rows, a_cols) = if op_a == wwr_blas_operation::WWR_BLAS_OP_N {
        (m, k)
    } else {
        (k, m)
    };
    check_extent("gemm_rows", "A", a.len(), a_rows, a_cols, lda);
    // Both blocks lie in the rows before the later one's end.
    check_extent("gemm_rows", "B", b.len(), (src + k).max(dst + m), n, ldb);
    let b = b.as_mut_ptr();
    // `wrapping_add`, as in split_at_mut: with n zero the slice may be empty.
    let (read, written) = (
        b.wrapping_add(src as usize).cast_const(),
        b.wrapping_add(dst as usize),
    );
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); b is its memory's only user
    // (DeviceSliceMut's), so apart from A, and within it the product reads
    // rows src to src + k and writes rows dst to dst + m of each column,
    // which the contract holds apart, so no element is both read and
    // written; the scalars are host values, the handle's default pointer
    // mode.
    gpu_status(unsafe {
        T::gemm_raw(
            handle.get(),
            op_a,
            wwr_blas_operation::WWR_BLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            a.as_ptr(),
            lda,
            read,
            ldb,
            &beta,
            written,
            ldb,
        )
    })
}

/// `op(A) x = b` on `handle`'s stream, x written over b: column-major, A
/// triangular n × n at its leading dimension, only its `fill` triangle read
/// (its diagonal taken as ones under `WWR_BLAS_DIAG_UNIT`), the vector n
/// entries at stride `incx`. wwr's `trsv<T>`, with both operands slices it
/// checks. It takes no workspace on either backend.
///
/// The borrow checker keeps x apart from A. The stride is at least 1: the
/// library's negative ones are refused. A singular A is not detected: x
/// comes back Inf or NaN.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If A's leading dimension is less than n (or 1), the stride is less than
/// 1, or a slice is shorter than its operand's extent; before anything is
/// enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, n), "n is not negative")]
#[track_caller]
pub fn trsv<T: BlasScalar>(
    handle: &BlasHandle,
    fill: wwr_blas_fill_mode,
    op: wwr_blas_operation,
    diag: wwr_blas_diag_type,
    n: i32,
    a: &impl DeviceSlice<T>,
    lda: i32,
    x: &mut impl DeviceSliceMut<T>,
    incx: i32,
) -> Status {
    check_extent("trsv", "A", a.len(), n, n, lda);
    check_vector("trsv", "x", x.len(), n, incx);
    let x = x.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); x is its memory's only user
    // (DeviceSliceMut's), so apart from A.
    gpu_status(unsafe { T::trsv_raw(handle.get(), fill, op, diag, n, a.as_ptr(), lda, x, incx) })
}

/// `x = alpha x` on `handle`'s stream: the vector n entries at stride
/// `incx`, the entries between them left as they were. wwr's `scal<T>`,
/// with x a slice it checks.
///
/// `alpha` is a host value, read in the handle's default pointer mode. The
/// stride is at least 1.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If the stride is less than 1, or the slice is shorter than the vector's
/// extent; before anything is enqueued.
#[requires(le!(0, n), "n is not negative")]
#[track_caller]
pub fn scal<T: BlasScalar>(
    handle: &BlasHandle,
    n: i32,
    alpha: T,
    x: &mut impl DeviceSliceMut<T>,
    incx: i32,
) -> Status {
    check_vector("scal", "x", x.len(), n, incx);
    let x = x.as_mut_ptr();
    // SAFETY: the handle is live; x is device memory covering its extent,
    // checked above, and stays so, with no other user, until the enqueued
    // work finishes (DeviceSliceMut's promise); alpha is a host value, the
    // handle's default pointer mode.
    gpu_status(unsafe { T::scal_raw(handle.get(), n, &alpha, x, incx) })
}

/// `C = alpha op(A) op(A)ᵀ + beta C` on `handle`'s stream, only C's `fill`
/// triangle read and written: column-major, `op(A)` n × k, so A stored
/// n × k under `WWR_BLAS_OP_N` and k × n under `_T`, C n × n, each at its
/// leading dimension. wwr's `syrk<T>`, with every matrix a slice it checks.
/// The transpose is plain on a complex `T` too: this is the symmetric rank-k
/// update, not the Hermitian one.
///
/// `alpha` and `beta` are host values, read in the handle's default pointer
/// mode. With `beta` zero, C's triangle need not be initialised; the other
/// triangle is neither read nor written. The borrow checker keeps C apart
/// from A; blocks of one matrix are its disjoint parts
/// ([`split_at_mut`](crate::device_slice::split_at_mut)).
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status (an
/// `op` of `WWR_BLAS_OP_C` on a complex `T` among them).
///
/// # Panics
/// If a leading dimension is less than its matrix's row count (or 1), or a
/// slice is shorter than its matrix at its leading dimension; before
/// anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, n) && le!(0, k), "n and k are not negative")]
#[track_caller]
pub fn syrk<T: BlasScalar>(
    handle: &BlasHandle,
    fill: wwr_blas_fill_mode,
    op: wwr_blas_operation,
    n: i32,
    k: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    beta: T,
    c: &mut impl DeviceSliceMut<T>,
    ldc: i32,
) -> Status {
    let (a_rows, a_cols) = if op == wwr_blas_operation::WWR_BLAS_OP_N {
        (n, k)
    } else {
        (k, n)
    };
    check_extent("syrk", "A", a.len(), a_rows, a_cols, lda);
    check_extent("syrk", "C", c.len(), n, n, ldc);
    let c = c.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); C is its memory's only user
    // (DeviceSliceMut's), so apart from A; the scalars are host values, the
    // handle's default pointer mode.
    gpu_status(unsafe {
        T::syrk_raw(
            handle.get(),
            fill,
            op,
            n,
            k,
            &alpha,
            a.as_ptr(),
            lda,
            &beta,
            c,
            ldc,
        )
    })
}

/// `C = alpha op_a(A) + beta op_b(B)` on `handle`'s stream: column-major,
/// C m × n at its leading dimension, A stored m × n under
/// `WWR_BLAS_OP_N` and n × m under `_T` or `_C` (the same as `_T` on a real
/// `T`), B likewise. wwr's `geam<T>`, with every matrix a slice it checks:
/// a sum, a scale, a copy (`beta` zero) or a transpose of a matrix, in
/// O(mn).
///
/// `alpha` and `beta` are host values, read in the handle's default pointer
/// mode. With `beta` zero B is still a slice covering its extent, since
/// the call is handed its pointer. The library also writes C over A or B;
/// here C is a slice of its own, which the borrow checker keeps apart from
/// both, so there is no in-place call.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If a leading dimension is less than its matrix's row count (or 1), or a
/// slice is shorter than its matrix at its leading dimension; before
/// anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n), "m and n are not negative")]
#[track_caller]
pub fn geam<T: BlasScalar>(
    handle: &BlasHandle,
    op_a: wwr_blas_operation,
    op_b: wwr_blas_operation,
    m: i32,
    n: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    beta: T,
    b: &impl DeviceSlice<T>,
    ldb: i32,
    c: &mut impl DeviceSliceMut<T>,
    ldc: i32,
) -> Status {
    let stored = |op| {
        if op == wwr_blas_operation::WWR_BLAS_OP_N {
            (m, n)
        } else {
            (n, m)
        }
    };
    let ((a_rows, a_cols), (b_rows, b_cols)) = (stored(op_a), stored(op_b));
    check_extent("geam", "A", a.len(), a_rows, a_cols, lda);
    check_extent("geam", "B", b.len(), b_rows, b_cols, ldb);
    check_extent("geam", "C", c.len(), m, n, ldc);
    let c = c.as_mut_ptr();
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, and stays so until the enqueued work finishes
    // (DeviceSlice's promise); C is its memory's only user
    // (DeviceSliceMut's), so apart from A and B; the scalars are host
    // values, the handle's default pointer mode.
    gpu_status(unsafe {
        T::geam_raw(
            handle.get(),
            op_a,
            op_b,
            m,
            n,
            &alpha,
            a.as_ptr(),
            lda,
            &beta,
            b.as_ptr(),
            ldb,
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
    #[allow(
        clippy::disallowed_methods,
        reason = "a test reads its result back on the host: the harness waits, not a task"
    )]
    fn binds_the_owners_stream() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        assert_eq!(blas.dev_idx(), 0);
        assert_eq!(blas.stream(), owner.stream());
        let mut bound: wwr_stream = std::ptr::null_mut();
        // SAFETY: the handle is live; writes one stream through a local.
        let status = unsafe { wwr_sys::wwr_blas_get_stream(blas.get(), &mut bound) };
        assert_eq!(status, wwr_blas_status::WWR_BLAS_STATUS_SUCCESS);
        assert_eq!(bound, owner.stream());
        drop(blas);
        assert_eq!(Arc::strong_count(&owner), 1);
        ok(owner.sync());
    }

    /// Needs a GPU. A handle handed a 32 MiB workspace (the size the
    /// planner gives each worker's) owns it, and runs a DGEMM through it to
    /// the exact product: small integers, so every partial sum is exact in
    /// any order. Handed a second, it owns that one, and still computes.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[allow(
        clippy::disallowed_methods,
        reason = "a test reads its result back on the host: the harness waits, not a task"
    )]
    fn runs_a_gemm_on_a_workspace_it_was_handed() {
        const N: usize = 16;
        const BYTES: usize = 32 << 20;
        let owner = Arc::new(Stream::new(0));
        let mut blas = BlasHandle::new(owner.clone());
        assert!(blas.workspace().is_none());
        let workspace = DeviceBytes::new(BYTES);
        let at = workspace.0 as usize;
        assert_eq!(at % 256, 0);
        blas.bind_workspace(workspace).unwrap();
        let held = blas.workspace().expect("a workspace");
        assert_eq!((held.as_ptr() as usize, held.len()), (at, BYTES));

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

        let second = DeviceBytes::new(BYTES);
        let at = second.0 as usize;
        blas.bind_workspace(second).unwrap();
        assert_eq!(blas.workspace().map(|w| w.as_ptr() as usize), Some(at));
        let mut dc = DeviceBytes::new(N * N * size_of::<f64>());
        gemm(
            &blas, OP_N, OP_N, n, n, n, 1.0, &da, n, &db, n, 0.0, &mut dc, n,
        )
        .unwrap();
        assert_eq!(download(&dc, &owner), want);
        drop(blas);
        ok(owner.sync());
    }

    /// Needs a GPU. A stream and the BLAS and solver handles bound to it,
    /// made on this thread, move to another and are used there: a DGEMM to
    /// the exact product (small integers, as above) and the solver's
    /// workspace query. The stream stays this thread's too, through its
    /// `Arc`, and the handles are destroyed on the other thread.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[allow(
        clippy::disallowed_methods,
        reason = "a test reads its result back on the host: the harness waits, not a task"
    )]
    fn a_stream_and_its_handles_move_to_another_thread() {
        use crate::solver::{SolverHandle, syevd_lwork};
        use wwr_sys::{wwr_solver_eig_mode, wwr_solver_fill_mode};

        const N: usize = 16;
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let solver = SolverHandle::new(owner.clone());
        let a: Vec<f64> = (0..N * N).map(|i| (i % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..N * N).map(|i| (i % 5) as f64 - 2.0).collect();
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

        let (got, lwork) = std::thread::spawn(move || {
            let n = N as i32;
            let mut dc = DeviceBytes::new(N * N * size_of::<f64>());
            gemm(
                &blas, OP_N, OP_N, n, n, n, 1.0, &da, n, &db, n, 0.0, &mut dc, n,
            )
            .unwrap();
            let lwork = syevd_lwork::<f64>(
                &solver,
                wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR,
                wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER,
                n,
                n,
            )
            .unwrap();
            (download(&dc, blas.owner()), lwork)
        })
        .join()
        .unwrap();

        assert_eq!(got, want);
        assert!(lwork > 0);
        assert_eq!(Arc::strong_count(&owner), 1);
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
        let owner = Arc::new(Stream::new(0));
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
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(16 * size_of::<f64>());
        let mut c = DeviceBytes::new(15 * size_of::<f64>());
        let _ = gemm(
            &blas, OP_N, OP_N, 4, 4, 4, 1.0, &a, 4, &a, 4, 0.0, &mut c, 4,
        );
    }

    /// Needs a GPU. Three products of transposed A's at a padded stride
    /// against one shared B (stride 0), into C's at a stride past their
    /// extent, beta non-zero: each product exact, and the elements between
    /// the C's left as they were.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemm_strided_batched_strides_each_operand_and_shares_a_zero_stride() {
        let (m, n, k, batch) = (4, 3, 5, 3);
        let (lda, ldb, ldc) = (k + 1, k + 2, m + 1);
        // A_i is stored k × m (op T) at stride lda m + 3; B, k × n, shared.
        let (stride_a, stride_c) = (lda * m + 3, ldc * n + 2);
        let a = matrix(stride_a * batch, 1, 4);
        let b = matrix(ldb, n, 5);
        let c = matrix(stride_c * batch, 1, 6);
        let mut want = c.clone();
        for i in 0..batch {
            let block = host_gemm(
                true,
                false,
                (m, n, k),
                2.0,
                (&a[i * stride_a..], lda),
                (&b, ldb),
                -1.0,
                (&c[i * stride_c..i * stride_c + ldc * n], ldc),
            );
            want[i * stride_c..i * stride_c + ldc * n].copy_from_slice(&block);
        }
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, db, mut dc) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
            DeviceBytes::upload(&c, &owner),
        );
        let i = |x: usize| x as i32;
        let l = |x: usize| x as i64;
        gemm_strided_batched(
            &blas,
            OP_T,
            OP_N,
            i(m),
            i(n),
            i(k),
            2.0,
            &da,
            i(lda),
            l(stride_a),
            &db,
            i(ldb),
            0,
            -1.0,
            &mut dc,
            i(ldc),
            l(stride_c),
            i(batch),
        )
        .unwrap();
        assert_eq!(download(&dc, &owner), want);
    }

    /// The last matrix of a batch ends inside its slice, and one more
    /// element is past it.
    #[test]
    fn a_batch_fits_to_its_last_matrix() {
        // Three 2 × 2 at ld 3 and stride 7: the last starts at 14, ends at 19.
        check_batch("t", "C", 19, 2, 2, 3, 7, 3, true);
        check_batch("t", "B", 5, 2, 2, 3, 0, 3, false);
    }

    #[test]
    #[should_panic(
        expected = "t: C is 3 matrices 2 x 2 at leading dimension 3 and stride 7, but its slice \
                    holds 18 elements"
    )]
    fn a_batch_one_past_the_slice_panics() {
        check_batch("t", "C", 18, 2, 2, 3, 7, 3, true);
    }

    /// Two products writing the same element: a C stride under one
    /// matrix's extent.
    #[test]
    #[should_panic(expected = "t: C's stride 4 is less than one matrix's 5 elements")]
    fn a_written_batch_that_overlaps_panics() {
        check_batch("t", "C", 100, 2, 2, 3, 4, 3, true);
    }

    #[test]
    #[should_panic(expected = "t: A's stride -1 is negative")]
    fn a_negative_batch_stride_panics() {
        check_batch("t", "A", 100, 2, 2, 3, -1, 3, false);
    }

    /// One gemv on a card, `y = -2 op(A) x + 3 y`, A's leading dimension
    /// padded and both vectors at stride 2: y's entries are the host's,
    /// exact on small integers, and the entries between them come back as
    /// they were.
    fn check_gemv(op: wwr_blas_operation) {
        let (m, n) = (5, 3);
        let lda = m + 2;
        let (x_count, y_count) = if op == OP_N { (n, m) } else { (m, n) };
        let (incx, incy) = (2, 2);
        let a = matrix(lda, n, 1);
        let x = matrix(1 + (x_count - 1) * incx, 1, 2);
        let y = matrix(1 + (y_count - 1) * incy, 1, 3);
        let mut want = y.clone();
        for r in 0..y_count {
            let dot: f64 = (0..x_count)
                .map(|l| {
                    let at = if op == OP_N {
                        a[r + l * lda]
                    } else {
                        a[l + r * lda]
                    };
                    at * x[l * incx]
                })
                .sum();
            want[r * incy] = -2.0 * dot + 3.0 * y[r * incy];
        }
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, dx, mut dy) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&x, &owner),
            DeviceBytes::upload(&y, &owner),
        );
        let i = |v: usize| v as i32;
        gemv::<f64>(
            &blas,
            op,
            i(m),
            i(n),
            -2.0,
            &da,
            i(lda),
            &dx,
            i(incx),
            3.0,
            &mut dy,
            i(incy),
        )
        .unwrap();
        assert_eq!(download(&dy, &owner), want);
    }

    /// Needs a GPU. `y = A x`, the fitted factor contracted with a density.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemv_multiplies_by_a() {
        check_gemv(OP_N);
    }

    /// Needs a GPU. `y = Aᵀ x`: x has m entries and y n.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemv_multiplies_by_the_transpose() {
        check_gemv(OP_T);
    }

    /// Needs a GPU. Under `OP_T` y has n entries, so a y of n - 1 is short
    /// and panics before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "gemv: y is 3 entries at stride 1, 3 elements, but its slice holds 2"
    )]
    fn gemv_refuses_a_short_slice() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(15 * size_of::<f64>());
        let x = DeviceBytes::new(5 * size_of::<f64>());
        let mut y = DeviceBytes::new(2 * size_of::<f64>());
        let _ = gemv::<f64>(&blas, OP_T, 5, 3, 1.0, &a, 5, &x, 1, 0.0, &mut y, 1);
    }

    use wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT as NON_UNIT;
    use wwr_blas_fill_mode::{
        WWR_BLAS_FILL_MODE_LOWER as LOWER, WWR_BLAS_FILL_MODE_UPPER as UPPER,
    };

    /// What an unread triangle or a padding row holds: a NaN, so a library
    /// that read it would carry it into the result.
    const UNREAD: f64 = f64::NAN;

    /// A k × k lower-triangular matrix at leading dimension `ld`: ones on
    /// the diagonal and -1, 0 or 1 below it, so its inverse has small
    /// integer entries and a solve against it is exact in any order and by
    /// any blocking; NaN above the diagonal and in the padding rows.
    fn unit_lower(k: usize, ld: usize) -> Vec<f64> {
        let mut a = vec![UNREAD; ld * k];
        for j in 0..k {
            a[j + j * ld] = 1.0;
            for i in j + 1..k {
                a[i + j * ld] = ((i * 3 + j) % 3) as f64 - 1.0;
            }
        }
        a
    }

    /// One trsv on a card after a scal, `op(A) x = 2 b`, with A's strict
    /// upper triangle and its padding rows NaN and the vector at stride 2:
    /// x is the exact solution, and the entries between its own come back
    /// as they were.
    fn check_trsv(op: wwr_blas_operation) {
        let (n, lda, inc) = (5, 7, 2);
        let a = unit_lower(n, lda);
        let x = matrix(n, 1, 4);
        let at = |i: usize, j: usize| {
            let (i, j) = if op == OP_N { (i, j) } else { (j, i) };
            if i >= j { a[i + j * lda] } else { 0.0 }
        };
        // b = op(A) x / 2 at stride 2, exact in halves; NaN between.
        let mut b = vec![UNREAD; 1 + (n - 1) * inc];
        for i in 0..n {
            b[i * inc] = (0..n).map(|l| at(i, l) * x[l]).sum::<f64>() / 2.0;
        }
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, mut db) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
        );
        scal::<f64>(&blas, n as i32, 2.0, &mut db, inc as i32).unwrap();
        trsv::<f64>(
            &blas, LOWER, op, NON_UNIT, n as i32, &da, lda as i32, &mut db, inc as i32,
        )
        .unwrap();
        let got = download(&db, &owner);
        for (r, g) in got.iter().enumerate() {
            if r % inc == 0 {
                assert_eq!(*g, x[r / inc], "x({})", r / inc);
            } else {
                assert!(g.is_nan(), "the entry between, {r}, is {g}");
            }
        }
    }

    /// Needs a GPU. `L x = 2 b`, one right-hand side of a left-sided solve.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn scal_then_trsv_solves_one_right_hand_side() {
        check_trsv(OP_N);
    }

    /// Needs a GPU. `Lᵀ x = 2 b`.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn scal_then_trsv_solves_against_the_transpose() {
        check_trsv(OP_T);
    }

    /// One gemm_rows on a card, `B(dst.., :) = -op(A) B(src.., :) + 2
    /// B(dst.., :)`, every leading dimension padded: the rows written are
    /// the host's, exact on small integers, and every other row of B comes
    /// back as it was.
    fn check_gemm_rows(op: wwr_blas_operation, src: usize, dst: usize) {
        let (m, n, k) = (3, 4, 2);
        let (ldb, lda) = (8, if op == OP_N { m + 1 } else { k + 1 });
        let a = matrix(lda, if op == OP_N { k } else { m }, 1);
        let b = matrix(ldb, n, 2);
        let mut want = b.clone();
        for j in 0..n {
            for i in 0..m {
                let dot: f64 = (0..k)
                    .map(|l| {
                        let at = if op == OP_N {
                            a[i + l * lda]
                        } else {
                            a[l + i * lda]
                        };
                        at * b[src + l + j * ldb]
                    })
                    .sum();
                want[dst + i + j * ldb] = -dot + 2.0 * b[dst + i + j * ldb];
            }
        }
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, mut db) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&b, &owner),
        );
        let i = |v: usize| v as i32;
        gemm_rows::<f64>(
            &blas,
            op,
            i(m),
            i(n),
            i(k),
            -1.0,
            &da,
            i(lda),
            &mut db,
            i(ldb),
            i(src),
            2.0,
            i(dst),
        )
        .unwrap();
        assert_eq!(download(&db, &owner), want);
    }

    /// Needs a GPU. The rows written below the rows read, a lower solve's
    /// update.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemm_rows_updates_the_rows_below() {
        check_gemm_rows(OP_N, 1, 3);
    }

    /// Needs a GPU. The rows written above the rows read, with A stored
    /// transposed: an upper solve's update.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn gemm_rows_updates_the_rows_above() {
        check_gemm_rows(OP_T, 4, 1);
    }

    /// Needs a GPU. Rows 1 to 3 read and rows 2 to 5 written share row 2,
    /// and the contract refuses them before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(expected = "the rows read and the rows written are apart")]
    fn gemm_rows_refuses_blocks_that_overlap() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(6 * size_of::<f64>());
        let mut b = DeviceBytes::new(32 * size_of::<f64>());
        let _ = gemm_rows::<f64>(&blas, OP_N, 3, 4, 2, 1.0, &a, 3, &mut b, 8, 1, 1.0, 2);
    }

    /// Needs a GPU. The later block ends at row 6, past a leading dimension
    /// of 5, and panics before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(expected = "gemm_rows: B's leading dimension 5 is less than max(1, 6 rows)")]
    fn gemm_rows_refuses_a_block_past_the_leading_dimension() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(6 * size_of::<f64>());
        let mut b = DeviceBytes::new(32 * size_of::<f64>());
        let _ = gemm_rows::<f64>(&blas, OP_N, 3, 4, 2, 1.0, &a, 3, &mut b, 5, 0, 1.0, 3);
    }

    /// Needs a GPU. A vector of 4 entries at stride 2 needs 7 elements, so
    /// a slice of 6 panics before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "trsv: x is 4 entries at stride 2, 7 elements, but its slice holds 6"
    )]
    fn trsv_refuses_a_short_slice() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(16 * size_of::<f64>());
        let mut x = DeviceBytes::new(6 * size_of::<f64>());
        let _ = trsv::<f64>(&blas, UPPER, OP_N, NON_UNIT, 4, &a, 4, &mut x, 2);
    }

    /// One syrk on a card, `C = -op(A) op(A)ᵀ + 2 C` on the lower triangle,
    /// with every leading dimension padded: the lower triangle is the
    /// host's, exact, and the strict upper triangle and the padding, NaN,
    /// come back as they were.
    fn check_syrk(op: wwr_blas_operation) {
        let (n, k) = (5, 3);
        let (a_rows, a_cols) = if op == OP_N { (n, k) } else { (k, n) };
        let (lda, ldc) = (a_rows + 2, n + 1);
        let a = matrix(lda, a_cols, 5);
        let mut c = matrix(ldc, n, 6);
        for j in 0..n {
            for i in 0..ldc {
                if i < j || i >= n {
                    c[i + j * ldc] = UNREAD;
                }
            }
        }
        let opa = |i: usize, l: usize| {
            if op == OP_N {
                a[i + l * lda]
            } else {
                a[l + i * lda]
            }
        };
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        let (da, mut dc) = (
            DeviceBytes::upload(&a, &owner),
            DeviceBytes::upload(&c, &owner),
        );
        let i = |v: usize| v as i32;
        syrk::<f64>(
            &blas,
            LOWER,
            op,
            i(n),
            i(k),
            -1.0,
            &da,
            i(lda),
            2.0,
            &mut dc,
            i(ldc),
        )
        .unwrap();
        let got = download(&dc, &owner);
        for j in 0..n {
            for r in 0..ldc {
                let g = got[r + j * ldc];
                if r >= j && r < n {
                    let dot: f64 = (0..k).map(|l| opa(r, l) * opa(j, l)).sum();
                    assert_eq!(g, 2.0 * c[r + j * ldc] - dot, "C({r}, {j})");
                } else {
                    assert!(g.is_nan(), "C's unwritten ({r}, {j}) is {g}");
                }
            }
        }
    }

    /// Needs a GPU. `C - A Aᵀ` on the lower triangle, a Cholesky's trailing
    /// update.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn syrk_updates_the_lower_triangle() {
        check_syrk(OP_N);
    }

    /// Needs a GPU. The same with A stored transposed, k × n.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn syrk_reads_a_transposed_a() {
        check_syrk(OP_T);
    }

    /// Needs a GPU. A C one element short of n × n panics before anything
    /// is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "syrk: C is 3 x 3 at leading dimension 3, 9 elements, but its slice holds 8"
    )]
    fn syrk_refuses_a_short_slice() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(6 * size_of::<f64>());
        let mut c = DeviceBytes::new(8 * size_of::<f64>());
        let _ = syrk::<f64>(&blas, LOWER, OP_T, 3, 2, 1.0, &a, 2, 0.0, &mut c, 3);
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
        let owner = Arc::new(Stream::new(0));
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
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(12 * size_of::<f64>());
        let x = DeviceBytes::new(3 * size_of::<f64>());
        let mut c = DeviceBytes::new(12 * size_of::<f64>());
        let _ = dgmm::<f64>(&blas, LEFT, 4, 3, &a, 4, &x, 1, &mut c, 4);
    }
    /// Needs a GPU. `C = alpha op_a(A) + beta op_b(B)` at every pairing of
    /// ops, a rectangular shape and every leading dimension padded: small
    /// integers, so each element is exact, and C's padding rows are left as
    /// they were.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn geam_reads_each_operand_at_its_op_and_leading_dimension() {
        let (m, n) = (5, 3);
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner.clone());
        for (op_a, op_b) in [(OP_N, OP_N), (OP_N, OP_T), (OP_T, OP_N), (OP_T, OP_T)] {
            let stored = |op| if op == OP_N { (m, n) } else { (n, m) };
            let ((a_rows, a_cols), (b_rows, b_cols)) = (stored(op_a), stored(op_b));
            let (lda, ldb, ldc) = (a_rows + 2, b_rows + 1, m + 3);
            let (a, b, c) = (
                matrix(lda, a_cols, 1),
                matrix(ldb, b_cols, 2),
                matrix(ldc, n, 3),
            );
            let at = |x: &[f64], ld: usize, op, i: usize, j: usize| {
                if op == OP_N {
                    x[i + j * ld]
                } else {
                    x[j + i * ld]
                }
            };
            let mut want = c.clone();
            for j in 0..n {
                for i in 0..m {
                    want[i + j * ldc] =
                        -2.0 * at(&a, lda, op_a, i, j) + 3.0 * at(&b, ldb, op_b, i, j);
                }
            }
            let (da, db, mut dc) = (
                DeviceBytes::upload(&a, &owner),
                DeviceBytes::upload(&b, &owner),
                DeviceBytes::upload(&c, &owner),
            );
            let i = |x: usize| x as i32;
            geam(
                &blas,
                op_a,
                op_b,
                i(m),
                i(n),
                -2.0,
                &da,
                i(lda),
                3.0,
                &db,
                i(ldb),
                &mut dc,
                i(ldc),
            )
            .unwrap();
            assert_eq!(download(&dc, &owner), want, "{op_a:?}, {op_b:?}");
        }
    }

    /// Needs a GPU. Under `OP_T` B is stored n × m, so a B of 4 × 2 at
    /// leading dimension 4 holds 8 elements and a slice of 7 is short: a
    /// panic before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "geam: B is 4 x 2 at leading dimension 4, 8 elements, but its slice holds 7"
    )]
    fn geam_refuses_a_short_slice() {
        let owner = Arc::new(Stream::new(0));
        let blas = BlasHandle::new(owner);
        let a = DeviceBytes::new(8 * size_of::<f64>());
        let b = DeviceBytes::new(7 * size_of::<f64>());
        let mut c = DeviceBytes::new(8 * size_of::<f64>());
        let _ = geam::<f64>(&blas, OP_N, OP_T, 2, 4, 1.0, &a, 2, 1.0, &b, 4, &mut c, 2);
    }
}
