//! A square device matrix made symmetric in place, on the GPU: the C++
//! `src/recipes/symmetrize`, whose README is the reasoning (why not geam,
//! the tiles).
//!
//! Two calls a task makes inside its `run`, on its stream, as it calls
//! `fb_core_wwr::blas::gemm`; neither is a task of its own. Each is one launch
//! of the kernels forked into `recipes/kernels/symmetrize/`, through
//! `fb-recipes-sys`'s safe wrappers, which hold the `unsafe` and check every
//! extent the launch trusts; this crate holds none.
//!
//! | C++ | Here |
//! |---|---|
//! | `symmetrize(res, n, a, lda)`, `A ← (A + Aᵀ) / 2` | [`symmetrize`]`(stream, n, a, lda)`, `fb_recipes_sys`'s, re-exported |
//! | `symmetrize(res, fill, n, a, lda)`, the triangle opposite `fill` ← the transpose of `fill`'s | [`symmetrize_from`]`(stream, fill, n, a, lda)`: one name per call, since Rust has no overloads |
//! | `DeviceResources &res`, its stream and device | a [`Stream`], whose device the caller's worker has current |
//! | `double *a`, `n` and `lda` trusted | a [`DeviceSliceMut`], asserted to hold the block in every build |
//! | `InvalidConfig` for a negative `n` | nothing: `n` is a `usize` |
//! | `InvalidConfig` for `lda < n` | a contract, and the launcher's refusal as an error in a release build |
//! | `InvalidConfig` for a `fill` neither lower nor upper | nothing: [`wwr_blas_fill_mode`] has no other value |
//! | `wwrblasFillMode_t`, so a syrk's fill mode passes straight through | [`wwr_blas_fill_mode`], for the same reason |
//! | `Status`, from `gpu_status` of the launch | [`Status`], from the launch status through `fb_kernel_abi::launch_status` |
//!
//! **Both results are exactly symmetric.** The mirror is a copy. The average
//! writes `(i, j)` and `(j, i)` from two threads adding the same pair in
//! opposite orders and halving, `(a + b) · ½` and `(b + a) · ½`, the same
//! bits: IEEE addition commutes, the halving is exact short of the subnormal
//! range, and a sum times a constant has no product to contract into an
//! FMA. So each element is the host's `(a + b) / 2`, bit for bit.

#![forbid(unsafe_code)]

use fb_core_error_handling::Status;
use fb_core_wwr::device_slice::DeviceSliceMut;
use fb_core_wwr::runtime::Stream;
pub use fb_recipes_sys::symmetrize;
use wwr_sys::wwr_blas_fill_mode;

/// Overwrites the strict triangle opposite `fill` of the column-major
/// `n × n` block of `a` (leading dimension `lda`) with the transpose of
/// `fill`'s, in place, on `stream`: what fills the other half of a syrk's
/// result. With `WWR_BLAS_FILL_MODE_LOWER`, `a(i, j) ← a(j, i)` for every
/// `i < j`; with `WWR_BLAS_FILL_MODE_UPPER`, for every `i > j`.
///
/// The `fill` triangle and the diagonal are not written, and no value of
/// the opposite triangle reaches the result, so it may hold anything, NaN
/// included. Rows past `n` are neither read nor written, and an `n` of 0
/// enqueues nothing. One launch, without synchronizing:
/// [`fb_recipes_sys::mirror_triangle`].
///
/// # Errors
/// The launch's failure, as `fb_kernel_abi::launch_status` maps it. A fault
/// inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `n` or `lda` does not fit an `i32`, or `a` is shorter than the block,
/// `lda (n - 1) + n` elements; before anything is enqueued.
#[track_caller]
pub fn symmetrize_from(
    stream: &Stream,
    fill: wwr_blas_fill_mode,
    n: usize,
    a: &mut impl DeviceSliceMut<f64>,
    lda: usize,
) -> Status {
    let from_lower = fill == wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
    fb_recipes_sys::mirror_triangle(stream, from_lower, n, a, lda)
}
