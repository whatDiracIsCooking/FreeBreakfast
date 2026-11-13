//! `libfb_integrals_kernels.so`'s C ABI: the launch wrappers rust/ has forked
//! from `src/integrals`, as raw FFI (`docs/gpu_layer.md`, "Layering").
//!
//! The bindings are generated from `integrals/kernels/fb_integrals.h` by
//! `devtools/rust-bindings.py` and committed; `fb-device-check` regenerates
//! and diffs them. Every function is `unsafe`: the C side trusts its device
//! pointers and lengths. A launch wrapper only enqueues on the stream it is
//! given; the workspace query is a plain host function, which needs no
//! device. A struct crosses by address, from host memory; the arrays it
//! points at are device memory.
//!
//! | C++ (`src/integrals`) | Here |
//! |---|---|
//! | `PrimitivePairClassView`, `PrimitivePairSource`, `PrimitivePairRecords` | [`fb_primitive_pair_class_view`], [`fb_primitive_pair_source`], [`fb_primitive_pair_records`], `#[repr(C)]`, by pointer |
//! | `mmd_order(stream, n_rec, src, key, key_alt, perm, perm_alt)` | [`fb_mmd_order`]`(stream, n_rec, src, key, perm, workspace, workspace_bytes)`, the two alternates one workspace of [`fb_mmd_order_workspace_bytes`] bytes |
//! | `mmd_records`, `mmd_bound` | [`fb_mmd_records`], [`fb_mmd_bound`] |
//! | `ordered_pair_bound` | [`fb_ordered_pair_bound`] |
//! | `overlap_class` | [`fb_overlap_class`] |
//! | `kPairBoundMaxMomentum`, `kRecordBoundMaxMomentum`, the overlap launcher's `kMaxL` | [`FB_PAIR_BOUND_MAX_MOMENTUM`], [`FB_RECORD_BOUND_MAX_MOMENTUM`], [`FB_OVERLAP_MAX_MOMENTUM`] |
//! | `wwrErrorInvalidValue` for a refusal | [`FB_LAUNCH_INVALID_ARGUMENT`] (a count, a null pointer, a short workspace) or [`FB_LAUNCH_UNSUPPORTED`] (no kernel for the class) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_integrals_abi_version`] and [`fb_integrals_backend`] against the bindings' |
//! | the other one-electron families, J, K and the rest | not forked yet |

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

/// Checks that the `libfb_integrals_kernels.so` this process loaded is the one
/// these bindings describe: its [`fb_integrals_abi_version`] against
/// [`FB_INTEGRALS_ABI_VERSION`], and its [`fb_integrals_backend`] against the backend
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
    let found = unsafe { (fb_integrals_abi_version(), fb_integrals_backend()) };
    let expected = match env!("FB_KERNELS_BACKEND") {
        "CUDA" => FB_BACKEND_CUDA,
        _ => FB_BACKEND_HIP,
    };
    check_abi(found, (FB_INTEGRALS_ABI_VERSION, expected))
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
        "libfb_integrals_kernels.so is ABI version {} built for {}, but fb-integrals-sys \
         expects ABI version {} built for {}: rebuild the library from this tree \
         for this backend (README.md, \"Build and test\")",
        found.0,
        name(found.1),
        expected.0,
        name(expected.1)
    ))
}

// The structs' layout, which mmd_launch.cu static_asserts with the same
// numbers: both sides follow the C rule over the same fields, and these catch
// a field added on one side only.
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<fb_primitive_pair_source>() == 96);
    assert!(offset_of!(fb_primitive_pair_source, bra) == 8);
    assert!(offset_of!(fb_primitive_pair_source, rank) == 88);
    assert!(size_of::<fb_primitive_pair_records>() == 136);
    assert!(offset_of!(fb_primitive_pair_records, rank_b) == 128);
    assert!(size_of::<fb_primitive_pair_class_view>() == 176);
    assert!(offset_of!(fb_primitive_pair_class_view, zeta) == 16);
    assert!(offset_of!(fb_primitive_pair_class_view, rank_b) == 168);
};

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
