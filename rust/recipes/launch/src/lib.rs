//! The safe calls over `fb-recipes-sys`'s launch wrappers, and the blocked
//! triangular solve: the half of the recipes' bindings that takes
//! `fb-core-wwr`'s slices and stream. The raw bindings, the tile constants
//! and the load check ([`check_library`]) are `fb-recipes-sys`'s, which the
//! v0.0.4 layer's `fb-core-device` reaches without this crate.
//!
//! | C++ (`src/recipes`) | Here |
//! |---|---|
//! | `trsm<T>(res, side, fill, op, diag, m, n, alpha, a, lda, b, ldb)` (`src/recipes/trsm`'s `.cppm` and `.cpp`), `T` `float` or `double` | [`trsm`]`(blas, side, fill, op, diag, m, n, alpha, a, lda, b, ldb)`, `T` a [`TrsmScalar`], `f32` or `f64` |
//! | its `DeviceResources &res`: `res.blas()`, `res.stream()`, and a `ScopedDeviceIndex` for the call | the [`BlasHandle`](fb_core_wwr::blas::BlasHandle) and the stream it is bound to, so the kernel and the BLAS calls cannot land on two streams; the caller has its device current, as for every BLAS call |
//! | its `InvalidConfig` for a `side`, `fill`, `op` or `diag` that is none of its values | nothing: the four are Rust enums |
//! | its `InvalidConfig` for a negative `m` or `n`, an `lda` below `max(1, k)` or an `ldb` below `max(1, m)`; `a` and `b` trusted | always-on contracts, on the slices' extents too; the borrow checker keeps B apart from A |
//! | its `wwr::gemm<T, int32_t>` with B and C rows of one matrix; `wwr::scal`, `wwr::trsv` | `fb_core_wwr::blas`'s `gemm_rows` on the left and `gemm` between the two sides of a cut on the right; its `scal` and `trsv` |
//!
//! [`check_library`]: fb_recipes_sys::check_library
//!
//! **The safe calls.** [`inverse_sqrt`], [`merge_sort`], [`merge_sort_keys`],
//! [`merge_sort_workspace_bytes`], [`symmetrize`], [`mirror_triangle`],
//! [`cholesky_diagonal`] and [`diis_solve`] are what a caller uses
//! (`docs/gpu_layer.md`, "The safe wrapper"), as `fb_core_wwr::blas::gemm`
//! is over the BLAS: device memory as [`DeviceSlice`]s, which the arena's
//! rows implement, each length the launch trusts checked against them in
//! every build, by an always-on contract or an `assert!`, and the launch
//! status a [`Status`] through [`fb_kernel_abi::launch_status`]. They are a
//! crate apart from the bindings because they take `fb-core-wwr`'s types,
//! which the v0.0.4 layer must not reach; this crate holds the `unsafe` of
//! the launches they make, and is never published.
//!
//! **The triangular solve.** [`trsm`] is BLAS's trsm, the tree's own and
//! never the library's: blocked, each diagonal block by
//! [`fb_trsm_diagonal_f64`] or [`fb_trsm_diagonal_f32`] and the update
//! between blocks a gemm, a plain call that enqueues on its BLAS handle's
//! stream and takes no workspace. It is the one call here that is more
//! than a launch, the port of `src/recipes/trsm`'s module above the
//! bridge, and it lives here rather than in a crate of its own because
//! one more crate over `fb-core-wwr` passes the reverse-dependents bar
//! (`docs/testing.md`, "Selection: the workspace graph is the selection
//! graph"); its module (`trsm.rs`) holds no `unsafe`.
//!
//! [`fb_trsm_diagonal_f64`]: fb_recipes_sys::fb_trsm_diagonal_f64
//! [`fb_trsm_diagonal_f32`]: fb_recipes_sys::fb_trsm_diagonal_f32

mod trsm;

pub use trsm::trsm;

use contracts::{debug_requires, requires};
use fb_core_error_handling::Status;
use fb_core_log::{eq, fits_i32, le};
use fb_core_wwr::blas::BlasScalar;
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::gpu_status_at;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_recipes_sys::*;

/// A launch status as a [`Status`] naming the wrapper: a launch error by
/// its `wwr_status` name, at the caller of the safe call.
#[track_caller]
fn launched(what: &str, status: fb_launch_status) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}

/// A count as the `u32` the C ABI takes. A public call whose count is a
/// parameter holds it to a `u32` by a contract first, so the panic here is
/// `sort`'s alone, on a slice's length.
#[track_caller]
fn count_u32(routine: &str, name: &str, count: usize) -> u32 {
    let Ok(count) = u32::try_from(count) else {
        panic!("{routine}: {name} is {count}, past the launch's u32 count");
    };
    count
}

/// Replaces each of the first `count` elements of `data` with `1 / sqrt(x)`,
/// in place, on `stream`, except an element at or below the floor, which
/// becomes 0: [`fb_inverse_sqrt`], over slices it checks. The rest of
/// `data` is left as it was; a `count` of 0 enqueues nothing.
///
/// The floor is `floor`, or `floor` times the first element of
/// `floor_scale` when one is given: a value on the device, such as one a
/// copy enqueued before this on `stream` wrote, so a floor known only from
/// earlier work is never read on the host. The product is one correctly
/// rounded multiplication, the bits of the host's `floor * scale`; the
/// floor should be non-negative.
///
/// The floor test comes before the square root, so the result is never
/// Inf, nor a NaN that was not one already.
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
#[requires(le!(count, data.len()), "count is within data")]
#[requires(le!(count, u32::MAX as usize), "count fits the launch's u32")]
#[requires(
    floor_scale.is_none_or(|s| !s.is_empty()),
    "floor_scale, when given, holds a value"
)]
#[debug_requires(floor >= 0.0, "the floor is non-negative")]
#[track_caller]
pub fn inverse_sqrt(
    stream: &Stream,
    count: usize,
    data: &mut impl DeviceSliceMut<f64>,
    floor: f64,
    floor_scale: Option<&dyn DeviceSlice<f64>>,
) -> Status {
    let scale = floor_scale.map_or(std::ptr::null(), |scale| scale.as_ptr());
    let count = count_u32("inverse_sqrt", "count", count);
    let data = data.as_mut_ptr();
    // SAFETY: the stream is live; data is device memory of at least count
    // doubles, by the contract, and stays so, with no other user, until the
    // enqueued work finishes (DeviceSliceMut's promise); scale is null or
    // device memory of at least one double, by the contract, with no writer
    // until then (DeviceSlice's promise).
    let status = unsafe { fb_inverse_sqrt(stream.stream(), count, data, floor, scale) };
    launched("fb_inverse_sqrt", status)
}

/// The bytes of the workspace [`merge_sort`] (`with_values`) or
/// [`merge_sort_keys`] needs for `n` keys: [`fb_merge_sort_workspace_bytes`].
/// Zero up to one tile (2816 keys). A host function: it needs no device.
#[requires(le!(n, u32::MAX as usize), "n fits the launch's u32")]
#[track_caller]
pub fn merge_sort_workspace_bytes(n: usize, with_values: bool) -> usize {
    let n = count_u32("merge_sort_workspace_bytes", "n", n);
    // SAFETY: a host function of its arguments alone.
    unsafe { fb_merge_sort_workspace_bytes(n, with_values) }
}

/// Sorts all of `keys` ascending in IEEE-754 total order, stably, on
/// `stream`, and applies the same permutation to `values`:
/// [`fb_merge_sort`], over slices it checks. The keys move as raw bits, so
/// -0.0 sorts before +0.0 and NaNs sort by sign and payload past the
/// infinities. The borrow checker keeps the three slices apart.
///
/// # Errors
/// The first failed launch, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside a kernel surfaces at the next synchronization.
///
/// The contracts are checked before anything is enqueued.
#[requires(eq!(values.len(), keys.len()), "values is as long as keys")]
#[requires(le!(keys.len(), u32::MAX as usize), "the key count fits the launch's u32")]
#[requires(
    le!(merge_sort_workspace_bytes(keys.len(), true), workspace.len()),
    "workspace holds merge_sort_workspace_bytes(keys.len(), true)"
)]
#[requires(
    merge_sort_workspace_bytes(keys.len(), true) == 0
        || (workspace.as_ptr() as usize).is_multiple_of(FB_DEVICE_ALIGNMENT),
    "workspace is FB_DEVICE_ALIGNMENT-aligned when it is used"
)]
#[track_caller]
pub fn merge_sort(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    values: &mut impl DeviceSliceMut<u32>,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    sort(stream, keys, values.as_mut_ptr(), workspace)
}

/// [`merge_sort`] with no values: all of `keys` sorted, on `stream`;
/// [`fb_merge_sort`] with a null `values`.
///
/// # Errors
/// As [`merge_sort`].
///
/// The contracts are checked before anything is enqueued.
#[requires(le!(keys.len(), u32::MAX as usize), "the key count fits the launch's u32")]
#[requires(
    le!(merge_sort_workspace_bytes(keys.len(), false), workspace.len()),
    "workspace holds merge_sort_workspace_bytes(keys.len(), false)"
)]
#[requires(
    merge_sort_workspace_bytes(keys.len(), false) == 0
        || (workspace.as_ptr() as usize).is_multiple_of(FB_DEVICE_ALIGNMENT),
    "workspace is FB_DEVICE_ALIGNMENT-aligned when it is used"
)]
#[track_caller]
pub fn merge_sort_keys(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    sort(stream, keys, std::ptr::null_mut(), workspace)
}

/// The one launch of [`merge_sort`] and [`merge_sort_keys`], `values` null
/// for the second or else as long as `keys`; every length, the workspace's
/// size and its alignment are the callers' contracts.
#[track_caller]
fn sort(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    values: *mut u32,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    let n = count_u32("merge_sort", "the key count", keys.len());
    let workspace_ptr = workspace.as_mut_ptr();
    let keys = keys.as_mut_ptr();
    // SAFETY: the stream is live; keys is device memory of n doubles, values
    // null or n u32s, and the workspace at least the query's bytes and
    // aligned as it partitions them, all by the callers' contracts; each
    // stays so, with no other user, until the enqueued work finishes
    // (DeviceSliceMut's promise), so none overlaps another.
    let status = unsafe {
        fb_merge_sort(
            stream.stream(),
            n,
            keys,
            values,
            workspace_ptr.cast(),
            workspace.len(),
        )
    };
    launched("fb_merge_sort", status)
}

/// `A ← (A + Aᵀ) / 2` in place over the column-major `n × n` block of `a`
/// with leading dimension `lda`, on `stream`: [`fb_symmetrize`], over a
/// slice it checks. `(i, j)` and `(j, i)` come out the same bits, since each
/// pair is summed once in each order, which IEEE addition makes the same
/// value, and halved; a symmetric input comes back unchanged. Rows past `n`
/// are neither read nor written, and an `n` of 0 enqueues nothing.
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `n` or `lda` does not fit an `i32`, or `a` is shorter than the block,
/// `lda (n - 1) + n` elements; before anything is enqueued.
// RELEASE: the launcher refuses an lda below n as FB_LAUNCH_INVALID_ARGUMENT
// before anything is touched.
#[debug_requires(lda >= n, "the leading dimension covers a column")]
#[track_caller]
pub fn symmetrize(
    stream: &Stream,
    n: usize,
    a: &mut impl DeviceSliceMut<f64>,
    lda: usize,
) -> Status {
    let (n, lda, a) = square_block("symmetrize", n, a, lda);
    // SAFETY: the stream is live; a holds the n × n block at leading
    // dimension lda, checked above, and stays so, with no other user, until
    // the enqueued work finishes (DeviceSliceMut's promise).
    let status = unsafe { fb_symmetrize(stream.stream(), n, a, lda) };
    launched("fb_symmetrize", status)
}

/// Overwrites the strict upper triangle of the column-major `n × n` block of
/// `a` (leading dimension `lda`) with the transpose of its strict lower one
/// when `from_lower`, or the strict lower with the transpose of the strict
/// upper when not, in place, on `stream`: [`fb_mirror_triangle`], over a
/// slice it checks. The source triangle and the diagonal are not written, and
/// no value of the destination reaches the result, so it may hold anything,
/// NaN included. Rows past `n` are neither read nor written, and an `n` of 0
/// enqueues nothing.
///
/// # Errors
/// As [`symmetrize`].
///
/// # Panics
/// As [`symmetrize`].
// RELEASE: as in symmetrize.
#[debug_requires(lda >= n, "the leading dimension covers a column")]
#[track_caller]
pub fn mirror_triangle(
    stream: &Stream,
    from_lower: bool,
    n: usize,
    a: &mut impl DeviceSliceMut<f64>,
    lda: usize,
) -> Status {
    let (n, lda, a) = square_block("mirror_triangle", n, a, lda);
    // SAFETY: as in symmetrize.
    let status = unsafe { fb_mirror_triangle(stream.stream(), from_lower, n, a, lda) };
    launched("fb_mirror_triangle", status)
}

/// The `n × n` block at leading dimension `lda` checked against `a`: `n` and
/// `lda` as the C ABI's `i32`s, and `a`'s pointer.
#[track_caller]
fn square_block(
    routine: &str,
    n: usize,
    a: &mut impl DeviceSliceMut<f64>,
    lda: usize,
) -> (i32, i32, *mut f64) {
    let (Ok(n32), Ok(lda32)) = (i32::try_from(n), i32::try_from(lda)) else {
        panic!("{routine}: n = {n} and lda = {lda} must each fit an i32");
    };
    // The last element the kernels touch is (n - 1, n - 1), at
    // lda (n - 1) + n - 1; with lda below n the launcher refuses before any
    // is touched.
    let need = if n == 0 { 0 } else { lda * (n - 1) + n };
    assert!(
        a.len() >= need,
        "{routine}: the {n} × {n} block at lda {lda} needs {need} elements, but a holds {}",
        a.len()
    );
    (n32, lda32, a.as_mut_ptr())
}

/// Factors the `jb × jb` diagonal tile at the start of `a` (column-major,
/// leading dimension `lda`) in place, `A = L Lᵀ` in its lower triangle, on
/// `stream`, and writes the same lower triangle into `l11`, packed (leading
/// dimension `jb`): [`fb_cholesky_diagonal`], over slices it checks. One
/// tile of a blocked factorization whose tile starts at column `first`.
///
/// `info` is the factorization's flag. If it is already non-zero when the
/// kernel runs, nothing is written; otherwise, at the tile's first pivot
/// that is not positive (or is NaN), column `k`, it becomes
/// `first + k + 1`, the 1-based leading minor of the whole matrix, and the
/// tile is left part-factored. The strict upper triangles of the tile and of
/// `l11` are neither read nor written. The borrow checker keeps the three
/// slices apart; the tile of a larger matrix is a part of it
/// ([`split_at_mut`](fb_core_wwr::device_slice::split_at_mut)).
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `first` or `lda` does not fit an `i32`, `a` is shorter than the tile,
/// `lda (jb - 1) + jb` elements, or `l11` shorter than `jb²`; before
/// anything is enqueued.
#[requires(!info.is_empty(), "info holds the flag")]
// RELEASE: the body clamps jb to one past FB_CHOLESKY_TILE, and the launcher
// refuses a jb outside 1 to FB_CHOLESKY_TILE as FB_LAUNCH_INVALID_ARGUMENT
// before anything is touched.
#[debug_requires(
    (1..=FB_CHOLESKY_TILE as usize).contains(&jb),
    "the tile's order is 1 to FB_CHOLESKY_TILE"
)]
// RELEASE: the launcher refuses an lda below jb as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(lda >= jb, "the leading dimension covers a column")]
#[track_caller]
pub fn cholesky_diagonal(
    stream: &Stream,
    jb: usize,
    first: usize,
    a: &mut impl DeviceSliceMut<f64>,
    lda: usize,
    l11: &mut impl DeviceSliceMut<f64>,
    info: &mut impl DeviceSliceMut<i32>,
) -> Status {
    let (Ok(first32), Ok(lda32)) = (i32::try_from(first), i32::try_from(lda)) else {
        panic!("cholesky_diagonal: first = {first} and lda = {lda} must each fit an i32");
    };
    // A jb past the tile is refused by the launcher before anything is
    // touched; clamped here only so the extents below cannot overflow.
    let jb = jb.min(FB_CHOLESKY_TILE as usize + 1);
    // The last element the kernel touches is (jb - 1, jb - 1), at
    // lda (jb - 1) + jb - 1; with lda below jb the launcher refuses first.
    let need = if jb == 0 { 0 } else { lda * (jb - 1) + jb };
    assert!(
        a.len() >= need,
        "cholesky_diagonal: the {jb} × {jb} tile at lda {lda} needs {need} elements, but a holds {}",
        a.len()
    );
    assert!(
        l11.len() >= jb * jb,
        "cholesky_diagonal: l11 holds {}, short of the {jb} × {jb} tile",
        l11.len()
    );
    let (a, l11, info) = (a.as_mut_ptr(), l11.as_mut_ptr(), info.as_mut_ptr());
    // SAFETY: the stream is live; a holds the tile at lda and l11 jb²
    // doubles, checked above, and info one flag, by the contract; each
    // stays so, with no other user, until the enqueued work finishes
    // (DeviceSliceMut's promise), so none overlaps another. jb fits an
    // i32: it is at most one past the tile.
    let status =
        unsafe { fb_cholesky_diagonal(stream.stream(), jb as i32, first32, a, lda32, l11, info) };
    launched("fb_cholesky_diagonal", status)
}

/// Solves the bordered DIIS system for the `m` extrapolation coefficients
/// on `stream`, in one block: [`fb_diis_solve`], over slices it checks.
///
/// ```text
/// [ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
/// [   -1^T      0 ] [ l ] = [ -1 ]
/// ```
///
/// `gram` is the residual Gram, column-indexed at leading dimension `ld`;
/// only its upper triangle over the first `m` columns, `gram[a ld + c]` for
/// `a <= c < m`, is read. The first `m` entries of `coeff` are written. A
/// pivot below `1e-14` makes the subspace singular: `coeff` is then the unit
/// selector at `newest`, so an extrapolation over it is the newest iterate,
/// and `singular[0]` is 1; otherwise 0.
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `gram` is shorter than the triangle it reads, `ld (m - 1) + m`
/// elements, or `coeff` shorter than `m`; before anything is enqueued.
#[requires(!singular.is_empty(), "singular holds the flag")]
// RELEASE: the launcher refuses an m of 0 as FB_LAUNCH_INVALID_ARGUMENT
// before reading anything.
#[debug_requires(m >= 1, "a subspace of at least one pair")]
// RELEASE: the body clamps m and ld to FB_DIIS_SOLVE_MAX_DIM, and the launcher
// refuses an ld below m as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(m <= ld, "the leading dimension covers the subspace")]
// RELEASE: the body clamps ld to FB_DIIS_SOLVE_MAX_DIM, and the launcher
// refuses it there as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(
    ld < FB_DIIS_SOLVE_MAX_DIM as usize,
    "the bordered system within the kernel's shared memory"
)]
// RELEASE: the body clamps newest to FB_DIIS_SOLVE_MAX_DIM, and the launcher
// refuses one at or past m as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(newest < m, "the newest pair is in the subspace")]
#[track_caller]
pub fn diis_solve(
    stream: &Stream,
    m: usize,
    ld: usize,
    newest: usize,
    gram: &impl DeviceSlice<f64>,
    coeff: &mut impl DeviceSliceMut<f64>,
    singular: &mut impl DeviceSliceMut<i32>,
) -> Status {
    // Past the system's largest order the launcher refuses before reading
    // anything; clamped here only so the extents below cannot overflow, and
    // so each fits the C ABI's i32.
    let bound = FB_DIIS_SOLVE_MAX_DIM as usize;
    let (m, ld, newest) = (m.min(bound), ld.min(bound), newest.min(bound));
    // The last element the kernel reads is (m - 1, m - 1), at
    // ld (m - 1) + m - 1; with ld below m the launcher refuses first.
    let need = if m == 0 { 0 } else { ld * (m - 1) + m };
    assert!(
        gram.len() >= need,
        "diis_solve: the Gram of {m} columns at ld {ld} needs {need} elements, but gram holds {}",
        gram.len()
    );
    assert!(
        coeff.len() >= m,
        "diis_solve: coeff holds {}, short of the {m} coefficients",
        coeff.len()
    );
    let (coeff, singular) = (coeff.as_mut_ptr(), singular.as_mut_ptr());
    // SAFETY: the stream is live; gram covers the triangle the kernel
    // reads and coeff m doubles, checked above, and singular one flag, by
    // the contract; each stays so until the enqueued work finishes
    // (DeviceSlice's and DeviceSliceMut's promises), the two written ones
    // apart from everything else. m, ld and newest are at most the bound, an i32.
    let status = unsafe {
        fb_diis_solve(
            stream.stream(),
            m as i32,
            ld as i32,
            newest as i32,
            gram.as_ptr(),
            coeff,
            singular,
        )
    };
    launched("fb_diis_solve", status)
}

mod sealed {
    pub trait Sealed {}
}

/// An element type [`trsm`] solves in: `f32` and `f64`, the C++ `real_fp`.
/// Its method is the diagonal block's launch wrapper of the type's suffix.
/// Sealed, since the safe call over it trusts the method to be that wrapper.
pub trait TrsmScalar:
    BlasScalar + PartialEq + std::ops::Neg<Output = Self> + sealed::Sealed
{
    /// The launch wrapper's name, for its status.
    #[doc(hidden)]
    const TRSM_DIAGONAL: &str;

    /// [`fb_trsm_diagonal_f64`] or [`fb_trsm_diagonal_f32`].
    ///
    /// # Safety
    /// The wrapper's: `stream` is live; `a` and `b` are device memory on its
    /// device covering the block and the right-hand sides at `lda` and
    /// `ldb`, apart from each other, alive until the enqueued work has
    /// finished, `b` with no other user.
    #[allow(clippy::too_many_arguments, reason = "the launch wrapper's signature")]
    unsafe fn trsm_diagonal_raw(
        stream: wwr_stream,
        jb: i32,
        ncols: i32,
        s: Self,
        a: *const Self,
        lda: i32,
        a_trans: bool,
        upper: bool,
        unit: bool,
        b: *mut Self,
        ldb: i32,
        b_trans: bool,
    ) -> fb_launch_status;
}

macro_rules! trsm_scalar {
    ($($t:ty => $launch:ident);* $(;)?) => {$(
        impl sealed::Sealed for $t {}

        impl TrsmScalar for $t {
            const TRSM_DIAGONAL: &str = stringify!($launch);

            unsafe fn trsm_diagonal_raw(
                stream: wwr_stream,
                jb: i32,
                ncols: i32,
                s: Self,
                a: *const Self,
                lda: i32,
                a_trans: bool,
                upper: bool,
                unit: bool,
                b: *mut Self,
                ldb: i32,
                b_trans: bool,
            ) -> fb_launch_status {
                // SAFETY: the caller's, which are the wrapper's.
                unsafe {
                    $launch(stream, jb, ncols, s, a, lda, a_trans, upper, unit, b, ldb, b_trans)
                }
            }
        }
    )*};
}

trsm_scalar! {
    f32 => fb_trsm_diagonal_f32;
    f64 => fb_trsm_diagonal_f64;
}

/// Solves `M X = s B` in place for one `jb × jb` triangular diagonal block
/// M and `ncols` right-hand sides, on `stream`: [`fb_trsm_diagonal_f64`] or
/// [`fb_trsm_diagonal_f32`], over slices it checks. One block of a blocked
/// triangular solve ([`trsm`]).
///
/// `M(p, q)` is `a[q + p lda]` when `a_trans`, else `a[p + q lda]`; it is
/// upper triangular when `upper`, else lower, and only that triangle is
/// read. With `unit` its diagonal is taken as 1 and not read. `B(r, c)` is
/// `b[c + r ldb]` when `b_trans`, else `b[r + c ldb]`, and X overwrites it;
/// nothing else of `b` is read or written. With `s` zero, X is zero and
/// neither M nor B is read (BLAS's `alpha = 0`). A singular M is not
/// detected: X comes back Inf or NaN. The borrow checker keeps `b` apart
/// from `a`; a block of a larger matrix is a part of it
/// ([`split_at`](fb_core_wwr::device_slice::split_at),
/// [`split_at_mut`](fb_core_wwr::device_slice::split_at_mut)).
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `a` is shorter than the block, `lda (jb - 1) + jb` elements, or `b`
/// shorter than the right-hand sides, `ldb (ncols - 1) + jb` elements, or
/// `ldb (jb - 1) + ncols` when `b_trans`; before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the launch wrapper's signature")]
#[requires(
    fits_i32!(ncols) && fits_i32!(lda) && fits_i32!(ldb),
    "ncols, lda and ldb fit the launch's i32"
)]
// RELEASE: the body clamps jb to one past FB_TRSM_TILE, and the launcher
// refuses a jb outside 1 to FB_TRSM_TILE as FB_LAUNCH_INVALID_ARGUMENT
// before anything is touched.
#[debug_requires(
    (1..=FB_TRSM_TILE as usize).contains(&jb),
    "the block's order is 1 to FB_TRSM_TILE"
)]
// RELEASE: the launcher refuses an ncols of 0 as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(ncols >= 1, "at least one right-hand side")]
// RELEASE: the launcher refuses an lda below jb as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(lda >= jb, "A's leading dimension covers a column")]
// RELEASE: the launcher refuses an ldb below jb, or below ncols when
// b_trans, as FB_LAUNCH_INVALID_ARGUMENT.
#[debug_requires(
    ldb >= if b_trans { ncols } else { jb },
    "B's leading dimension covers a column"
)]
#[track_caller]
pub(crate) fn trsm_diagonal<T: TrsmScalar>(
    stream: &Stream,
    jb: usize,
    ncols: usize,
    s: T,
    a: &impl DeviceSlice<T>,
    lda: usize,
    a_trans: bool,
    upper: bool,
    unit: bool,
    b: &mut impl DeviceSliceMut<T>,
    ldb: usize,
    b_trans: bool,
) -> Status {
    // A jb past the block is refused by the launcher before anything is
    // touched; clamped here only so the extents below cannot overflow.
    let jb = jb.min(FB_TRSM_TILE as usize + 1);
    // The last element of M is (jb - 1, jb - 1), at lda (jb - 1) + jb - 1,
    // and of B (jb - 1, ncols - 1), at ldb (ncols - 1) + jb - 1, or at
    // ldb (jb - 1) + ncols - 1 transposed; with a leading dimension below
    // its column the launcher refuses first.
    let (need_a, need_b) = match (jb, ncols) {
        (0, _) | (_, 0) => (0, 0),
        _ if b_trans => (lda * (jb - 1) + jb, ldb * (jb - 1) + ncols),
        _ => (lda * (jb - 1) + jb, ldb * (ncols - 1) + jb),
    };
    assert!(
        a.len() >= need_a,
        "trsm_diagonal: the {jb} × {jb} block at lda {lda} needs {need_a} elements, but a holds {}",
        a.len()
    );
    assert!(
        b.len() >= need_b,
        "trsm_diagonal: {ncols} right-hand sides of {jb} rows at ldb {ldb} need {need_b} \
         elements, but b holds {}",
        b.len()
    );
    let b = b.as_mut_ptr();
    // SAFETY: the stream is live; a holds the block at lda and b the
    // right-hand sides at ldb, checked above; each stays so until the
    // enqueued work finishes, b with no other user (DeviceSlice's and
    // DeviceSliceMut's promises), so the two are apart. jb is at most one
    // past the block, and ncols, lda and ldb fit an i32 by the contract.
    let status = unsafe {
        T::trsm_diagonal_raw(
            stream.stream(),
            jb as i32,
            ncols as i32,
            s,
            a.as_ptr(),
            lda as i32,
            a_trans,
            upper,
            unit,
            b,
            ldb as i32,
            b_trans,
        )
    };
    launched(T::TRSM_DIAGONAL, status)
}
