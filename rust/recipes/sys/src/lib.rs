//! `libfb_recipes_kernels.so`'s C ABI: the launch wrappers rust/ has forked
//! from `src/recipes`, as raw FFI (`docs/gpu_layer.md`, "Layering").
//!
//! The bindings are generated from `recipes/kernels/fb_recipes.h` by
//! `devtools/rust-bindings.py` and committed; `fb-device-check` regenerates
//! and diffs them. Every function is `unsafe`: the C side trusts its device
//! pointers and lengths. A launch wrapper only enqueues on the stream it is
//! given; the workspace query is a plain host function, which needs no
//! device.
//!
//! | C++ (`src/recipes`) | Here |
//! |---|---|
//! | `merge_sort_passes(stream, n, keys, keys_alt, values, values_alt)` | [`fb_merge_sort`]`(stream, n, keys, values, workspace, workspace_bytes)` |
//! | `keys_alt` and `values_alt`, "n elements" each | one workspace of [`fb_merge_sort_workspace_bytes`] bytes, partitioned at [`FB_DEVICE_ALIGNMENT`] |
//! | `inverse_sqrt<T>(stream, count, data, floor)` (`src/recipes/orthogonalize`), through wwr's `parallel_for` | [`fb_inverse_sqrt`]`(stream, count, data, floor)`, `f64` only, its index-per-thread launch written out |
//! | `symmetrize(stream, n, a, lda)` and `mirror_triangle(stream, from_lower, n, a, lda)` (`src/recipes/symmetrize`), `lda` an `int64_t` | [`fb_symmetrize`] and [`fb_mirror_triangle`], `lda` the module's `int32_t`, the module's argument checks moved in as refusals |
//! | `cholesky_diagonal_block(stream, jb, first, a, lda, info)` (`src/recipes/cholesky`), `lda` an `int64_t`, `kCholeskyBlock` | [`fb_cholesky_diagonal`]`(stream, jb, first, a, lda, l11, info)`, `lda` the module's `int32_t`, the factored tile also written packed into `l11` for the panel's trsm, null pointers refused too; [`FB_CHOLESKY_TILE`] |
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_recipes_abi_version`] and [`fb_recipes_backend`] against the bindings' |
//! | `gather_columns` and every other recipe | not forked yet |
//!
//! **The safe calls.** [`inverse_sqrt`], [`merge_sort`], [`merge_sort_keys`],
//! [`merge_sort_workspace_bytes`], [`symmetrize`], [`mirror_triangle`] and
//! [`cholesky_diagonal`] are what a caller uses
//! (`docs/gpu_layer.md`, "The safe wrapper"), as `fb_core_wwr::blas::gemm`
//! is over the BLAS: device memory as
//! [`DeviceSlice`](fb_core_wwr::device_slice::DeviceSlice)s, which the
//! arena's rows implement, each length the launch trusts asserted against
//! them in every build, and the launch status a [`Status`] through
//! [`fb_kernel_abi::launch_status`]. They live beside the generated
//! bindings, not in a crate of their own: this crate is already where the
//! library's `unsafe` is allowed, and it is never published.

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

use contracts::debug_requires;
use fb_core_error_handling::Status;
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::device_slice::DeviceSliceMut;
use fb_core_wwr::gpu_status_at;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;

/// Checks that the `libfb_recipes_kernels.so` this process loaded is the one
/// these bindings describe: its [`fb_recipes_abi_version`] against
/// [`FB_RECIPES_ABI_VERSION`], and its [`fb_recipes_backend`] against the backend
/// of the CMake build this crate linked (`docs/gpu_layer.md`, "What the port
/// changes"). Call it once, before the first launch: a library built for the
/// other backend, or from another header, otherwise fails at a launch or deep
/// inside wwr with nothing naming the cause. Needs no device. The comparison
/// is [`fb_kernel_abi::check`], tested where no library is built.
///
/// # Errors
///
/// A mismatch, with a message naming the found and the expected values.
pub fn check_library() -> Result<(), String> {
    // SAFETY: a host function with no arguments, which calls no runtime.
    let version = unsafe { fb_recipes_abi_version() };
    // SAFETY: as above.
    let backend = unsafe { fb_recipes_backend() };
    let found = (version, backend);
    fb_kernel_abi::check(
        "recipes",
        found,
        (FB_RECIPES_ABI_VERSION, env!("FB_KERNELS_BACKEND")),
    )
}

// The numbers fb_kernel_abi::check names the found backend by.
const _: () = assert!(FB_BACKEND_CUDA == 1 && FB_BACKEND_HIP == 2);
// The numbers fb_kernel_abi::launch_status reads a launch status by.
const _: () = assert!(
    FB_LAUNCH_SUCCESS == 0 && FB_LAUNCH_UNSUPPORTED == -1 && FB_LAUNCH_INVALID_ARGUMENT == -2
);

/// A launch status as a [`Status`] naming the wrapper: a launch error by
/// its `wwr_status` name, at the caller of the safe call.
#[track_caller]
fn launched(what: &str, status: fb_launch_status) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}

/// A count as the `u32` the C ABI takes.
#[track_caller]
fn count_u32(routine: &str, name: &str, count: usize) -> u32 {
    let Ok(count) = u32::try_from(count) else {
        panic!("{routine}: {name} is {count}, past the launch's u32 count");
    };
    count
}

/// Replaces each of the first `count` elements of `data` with `1 / sqrt(x)`,
/// in place, on `stream`, except an element at or below `floor`, which
/// becomes 0: [`fb_inverse_sqrt`], over a slice it checks. The rest of
/// `data` is left as it was; a `count` of 0 enqueues nothing.
///
/// The floor test comes before the square root, so the result is never
/// Inf, nor a NaN that was not one already.
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it. A
/// fault inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `count` is more than `data` holds, or does not fit a `u32`; before
/// anything is enqueued.
#[debug_requires(floor >= 0.0, "the floor is non-negative")]
#[track_caller]
pub fn inverse_sqrt(
    stream: &Stream,
    count: usize,
    data: &mut impl DeviceSliceMut<f64>,
    floor: f64,
) -> Status {
    assert!(
        count <= data.len(),
        "inverse_sqrt: count is {count}, but data holds {}",
        data.len()
    );
    let count = count_u32("inverse_sqrt", "count", count);
    let data = data.as_mut_ptr();
    // SAFETY: the stream is live; data is device memory of at least count
    // doubles, checked above, and stays so, with no other user, until the
    // enqueued work finishes (DeviceSliceMut's promise).
    let status = unsafe { fb_inverse_sqrt(stream.stream(), count, data, floor) };
    launched("fb_inverse_sqrt", status)
}

/// The bytes of the workspace [`merge_sort`] (`with_values`) or
/// [`merge_sort_keys`] needs for `n` keys: [`fb_merge_sort_workspace_bytes`].
/// Zero up to one tile (2816 keys). A host function: it needs no device.
///
/// # Panics
/// If `n` does not fit a `u32`.
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
/// # Panics
/// If `values` is not as long as `keys`, the key count does not fit a
/// `u32`, or `workspace` is shorter than
/// [`merge_sort_workspace_bytes`]`(keys.len(), true)` or, when that is not
/// zero, not [`FB_DEVICE_ALIGNMENT`]-aligned; before anything is enqueued.
#[track_caller]
pub fn merge_sort(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    values: &mut impl DeviceSliceMut<u32>,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    assert!(
        values.len() == keys.len(),
        "merge_sort: values holds {}, but keys {}",
        values.len(),
        keys.len()
    );
    sort(stream, keys, values.as_mut_ptr(), workspace)
}

/// [`merge_sort`] with no values: all of `keys` sorted, on `stream`;
/// [`fb_merge_sort`] with a null `values`.
///
/// # Errors
/// As [`merge_sort`].
///
/// # Panics
/// If the key count does not fit a `u32`, or `workspace` is shorter than
/// [`merge_sort_workspace_bytes`]`(keys.len(), false)` or, when that is not
/// zero, not [`FB_DEVICE_ALIGNMENT`]-aligned; before anything is enqueued.
#[track_caller]
pub fn merge_sort_keys(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    sort(stream, keys, std::ptr::null_mut(), workspace)
}

/// The one launch of [`merge_sort`] and [`merge_sort_keys`], `values` null
/// for the second or else as long as `keys`, checked by the caller.
#[track_caller]
fn sort(
    stream: &Stream,
    keys: &mut impl DeviceSliceMut<f64>,
    values: *mut u32,
    workspace: &mut impl DeviceSliceMut<u8>,
) -> Status {
    let n = count_u32("merge_sort", "the key count", keys.len());
    // SAFETY: a host function of its arguments alone.
    let need = unsafe { fb_merge_sort_workspace_bytes(n, !values.is_null()) };
    assert!(
        workspace.len() >= need,
        "merge_sort: {n} keys need a workspace of {need} bytes, but it holds {}",
        workspace.len()
    );
    let workspace_ptr = workspace.as_mut_ptr();
    assert!(
        need == 0 || (workspace_ptr as usize).is_multiple_of(FB_DEVICE_ALIGNMENT),
        "merge_sort: the workspace at {workspace_ptr:p} is not {FB_DEVICE_ALIGNMENT}-byte aligned"
    );
    let keys = keys.as_mut_ptr();
    // SAFETY: the stream is live; keys is device memory of n doubles, values
    // null or n u32s, and the workspace at least the query's bytes and
    // aligned as it partitions them, all checked here or by the caller; each
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
/// `lda (jb - 1) + jb` elements, `l11` shorter than `jb²`, or `info` empty;
/// before anything is enqueued.
#[debug_requires(
    (1..=FB_CHOLESKY_TILE as usize).contains(&jb),
    "the tile's order is 1 to FB_CHOLESKY_TILE"
)]
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
    assert!(!info.is_empty(), "cholesky_diagonal: info is empty");
    let (a, l11, info) = (a.as_mut_ptr(), l11.as_mut_ptr(), info.as_mut_ptr());
    // SAFETY: the stream is live; a holds the tile at lda, l11 jb² doubles
    // and info one flag, each checked above, and each stays so, with no
    // other user, until the enqueued work finishes (DeviceSliceMut's
    // promise), so none overlaps another. jb fits an i32: it is at most one
    // past the tile.
    let status =
        unsafe { fb_cholesky_diagonal(stream.stream(), jb as i32, first32, a, lda32, l11, info) };
    launched("fb_cholesky_diagonal", status)
}
