//! `libfb_recipes_kernels.so`'s C ABI: the launch wrappers rust/ has forked
//! from `src/recipes`, as raw FFI (`docs/gpu_layer.md`, "Layering"), and
//! the library's load check.
//!
//! The bindings are generated from `recipes/kernels/fb_recipes.h` by
//! `devtools/rust-bindings.py` and committed; `fb-device-check` regenerates
//! and diffs them. Every function is `unsafe`: the C side trusts its device
//! pointers and lengths. A launch wrapper only enqueues on the stream it is
//! given; the workspace query is a plain host function, which needs no
//! device.
//!
//! **The raw half alone.** The safe calls over these bindings, which take
//! `fb-core-wwr`'s slices and stream, are `fb-recipes-launch`'s; the v0.0.4
//! layer's `fb-core-device` calls the bindings here directly, over its own
//! slices, and so reaches no type of `fb-core-wwr` (`docs/gpu_layer.md`,
//! "The v0.0.4 runtime"). This crate is the one that declares the library's
//! `links` key, so a change under `rust/recipes/kernels/` selects it and
//! every crate above it.
//!
//! | C++ (`src/recipes`) | Here |
//! |---|---|
//! | `merge_sort_passes(stream, n, keys, keys_alt, values, values_alt)` | [`fb_merge_sort`]`(stream, n, keys, values, workspace, workspace_bytes)` |
//! | `keys_alt` and `values_alt`, "n elements" each | one workspace of [`fb_merge_sort_workspace_bytes`] bytes, partitioned at [`FB_DEVICE_ALIGNMENT`] |
//! | `inverse_sqrt<T>(stream, count, data, floor)` (`src/recipes/orthogonalize`), through wwr's `parallel_for` | [`fb_inverse_sqrt`]`(stream, count, data, floor, floor_scale)`, `f64` only, its index-per-thread launch written out, and the floor optionally scaled by a value on the device |
//! | `symmetrize(stream, n, a, lda)` and `mirror_triangle(stream, from_lower, n, a, lda)` (`src/recipes/symmetrize`), `lda` an `int64_t` | [`fb_symmetrize`] and [`fb_mirror_triangle`], `lda` the module's `int32_t`, the module's argument checks moved in as refusals |
//! | `cholesky_diagonal_block(stream, jb, first, a, lda, info)` (`src/recipes/cholesky`), `lda` an `int64_t`, `kCholeskyBlock` | [`fb_cholesky_diagonal`]`(stream, jb, first, a, lda, l11, info)`, `lda` the module's `int32_t`, the factored tile also written packed into `l11` for the panel's trsm, null pointers refused too; [`FB_CHOLESKY_TILE`] |
//! | `diis_solve(stream, m, ld, newest, gram, coeff, singular_out)` (`src/recipes/diis`), `kDiisSolveMaxDim` | [`fb_diis_solve`], the C++ doc comment's bounds and null pointers refused; [`FB_DIIS_SOLVE_MAX_DIM`] |
//! | `trsm_diagonal_block(stream, jb, ncols, s, a, lda, a_trans, upper, unit, b, ldb, b_trans)` (`src/recipes/trsm`), overloaded on `float` and `double`, `lda` and `ldb` `int64_t`s, `kTrsmBlock` | [`fb_trsm_diagonal_f32`] and [`fb_trsm_diagonal_f64`], one entry per element type, `lda` and `ldb` the module's `int32_t`, a leading dimension under its column and null pointers refused too; [`FB_TRSM_TILE`] |
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_recipes_abi_version`] and [`fb_recipes_backend`] against the bindings' |
//! | `gather_columns` and every other recipe | not forked yet |
//!

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

use fb_kernel_abi::Backend;

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
    let built_for = const { Backend::named(env!("FB_KERNELS_BACKEND")) };
    fb_kernel_abi::check(
        "recipes",
        (version, backend),
        (FB_RECIPES_ABI_VERSION, built_for),
    )
}
