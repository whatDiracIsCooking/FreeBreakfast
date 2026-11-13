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
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | `gather_columns` and every other recipe | not forked yet |

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;
