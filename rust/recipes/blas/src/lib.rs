//! BLAS conveniences over [`fb_core_wwr::blas`]'s checked calls: the C++
//! `src/recipes/blas`, whose README is the reasoning.
//!
//! The general calls stay [`fb_core_wwr::blas`]'s ([`blas::gemm`], every
//! leading dimension and op passed in); this crate is the shorthand for the
//! common case, and holds no `unsafe`. Every matrix is a
//! [`DeviceSlice`], checked against its extent by the call underneath.
//!
//! | C++ | Here |
//! |---|---|
//! | `template <wwrblasOperation_t OpA, OpB>`, each `WWRBLAS_OP_N` when not given | type parameters, the markers [`N`], [`T`] and [`C`] behind [`BlasOp`]: `gemm::<T, N, _>`, both always given (a function has no default type parameter) |
//! | `concept BlasOp<Op>` | the trait [`BlasOp`] |
//! | `gemm<OpA, OpB>(handle, m, n, k, alpha, a, b, beta, c)`, packed | [`gemm`], packed |
//! | `gemm<OpA, OpB>(handle, n, alpha, a, b, beta, c)`, every matrix n × n | [`gemm`] at `m = n = k`: one name, not two |
//! | `alpha`, `beta` by pointer, as the handle's pointer mode says | by value, read in the default (host) mode |
//! | `handle`, a `BlasHandle` or a raw `wwrblasHandle_t` | a [`BlasHandle`] |
//! | a raw `wwrblasStatus_t`, the caller picking the tier | a [`Status`] |
//! | `gemmwich`, `gmmwich` | not ported yet |
//!
//! [`blas::gemm`]: fb_core_wwr::blas::gemm

#![forbid(unsafe_code)]

use fb_core_error_handling::Status;
use fb_core_wwr::blas::{self, BlasHandle, BlasScalar};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use wwr_sys::wwr_blas_operation;

/// A BLAS operation fixed at compile time: [`N`], [`T`] or [`C`]. The C++
/// `BlasOp` concept, over a non-type template parameter Rust's stable
/// const generics cannot take (an enum).
pub trait BlasOp {
    /// The operation, as the library takes it.
    const RAW: wwr_blas_operation;
}

/// `op(X) = X`.
pub struct N;
/// `op(X) = Xᵀ`.
pub struct T;
/// `op(X) = Xᴴ`, the conjugate transpose: the same as [`T`] on a real type.
pub struct C;

impl BlasOp for N {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
}
impl BlasOp for T {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
}
impl BlasOp for C {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_C;
}

/// `C = alpha OpA(A) OpB(B) + beta C` on `handle`'s stream, for
/// `OpA(A)` m × k, `OpB(B)` k × n and C m × n, every matrix column-major
/// and packed: A is stored m × k under [`N`] and k × m otherwise (its
/// leading dimension m or k), B k × n or n × k (k or n), and C's leading
/// dimension is m. [`blas::gemm`] with those leading dimensions; for a
/// padded one (a block of a larger matrix) or an op known only at run
/// time, call that.
///
/// `alpha` and `beta` are host values. With `beta` zero, C need not be
/// initialised. A leading dimension is at least 1, as the library asks, so
/// an empty product is a call that touches nothing.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// As [`blas::gemm`]: a negative dimension, or a slice shorter than its
/// matrix; before anything is enqueued.
///
/// [`blas::gemm`]: fb_core_wwr::blas::gemm
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[track_caller]
pub fn gemm<OpA: BlasOp, OpB: BlasOp, S: BlasScalar>(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    k: i32,
    alpha: S,
    a: &impl DeviceSlice<S>,
    b: &impl DeviceSlice<S>,
    beta: S,
    c: &mut impl DeviceSliceMut<S>,
) -> Status {
    let is_n = |op| op == wwr_blas_operation::WWR_BLAS_OP_N;
    let lda = if is_n(OpA::RAW) { m } else { k };
    let ldb = if is_n(OpB::RAW) { k } else { n };
    blas::gemm(
        handle,
        OpA::RAW,
        OpB::RAW,
        m,
        n,
        k,
        alpha,
        a,
        lda.max(1),
        b,
        ldb.max(1),
        beta,
        c,
        m.max(1),
    )
}
