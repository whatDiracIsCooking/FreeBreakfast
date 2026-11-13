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
/// inside wwr with nothing naming the cause. Needs no device.
///
/// # Errors
///
/// A mismatch, with a message naming the found and the expected values.
pub fn check_library() -> Result<(), String> {
    // SAFETY: two host functions with no arguments, which call no runtime.
    let found = unsafe { (fb_recipes_abi_version(), fb_recipes_backend()) };
    let expected = match env!("FB_KERNELS_BACKEND") {
        "CUDA" => FB_BACKEND_CUDA,
        _ => FB_BACKEND_HIP,
    };
    check_abi(found, (FB_RECIPES_ABI_VERSION, expected))
}

/// The load check on values passed in, `(version, backend)` found and
/// expected, so that a mismatch is testable with one build.
fn check_abi(found: (u32, fb_backend), expected: (u32, fb_backend)) -> Result<(), String> {
    let name = |b: fb_backend| match b {
        FB_BACKEND_CUDA => "CUDA".to_owned(),
        FB_BACKEND_HIP => "HIP".to_owned(),
        other => format!("unknown backend {other}"),
    };
    if found == expected {
        return Ok(());
    }
    Err(format!(
        "libfb_recipes_kernels.so is ABI version {} built for {}, but fb-recipes-sys \
         expects ABI version {} built for {}: rebuild the library from this tree \
         for this backend (README.md, \"Build and test\")",
        found.0,
        name(found.1),
        expected.0,
        name(expected.1)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_check_passes_equal_values() {
        assert_eq!(check_abi((3, FB_BACKEND_HIP), (3, FB_BACKEND_HIP)), Ok(()));
    }

    #[test]
    fn the_check_refuses_another_version_naming_both() {
        let e = check_abi((2, FB_BACKEND_CUDA), (3, FB_BACKEND_CUDA)).unwrap_err();
        assert!(e.contains("ABI version 2 built for CUDA"), "{e}");
        assert!(e.contains("expects ABI version 3 built for CUDA"), "{e}");
    }

    #[test]
    fn the_check_refuses_another_backend_naming_both() {
        let e = check_abi((3, FB_BACKEND_CUDA), (3, FB_BACKEND_HIP)).unwrap_err();
        assert!(e.contains("ABI version 3 built for CUDA"), "{e}");
        assert!(e.contains("expects ABI version 3 built for HIP"), "{e}");
        let e = check_abi((3, 7), (3, FB_BACKEND_HIP)).unwrap_err();
        assert!(e.contains("built for unknown backend 7"), "{e}");
    }
}
