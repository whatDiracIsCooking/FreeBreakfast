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
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_recipes_abi_version`] and [`fb_recipes_backend`] against the bindings' |
//! | `gather_columns` and every other recipe | not forked yet |
//!
//! **The safe calls.** [`inverse_sqrt`], [`merge_sort`], [`merge_sort_keys`]
//! and [`merge_sort_workspace_bytes`] are what a caller uses
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

/// [`merge_sort`] with no values: all of `keys` sorted, on `stream`.
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
