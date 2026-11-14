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

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

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
