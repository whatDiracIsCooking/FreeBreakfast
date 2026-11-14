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
//! | `kinetic_class` | [`fb_kinetic_class`] |
//! | `NucleiView` (`src/molecule`) | [`fb_nuclei_view`], `#[repr(C)]`, by pointer |
//! | `potential_en_class(stream, cls, nuclei, n_ao, v)` | [`fb_potential_en_class`]`(stream, cls, nuclei, n_ao, v)`, the nuclei by pointer |
//! | `kPairBoundMaxMomentum`, `kRecordBoundMaxMomentum`, the overlap, kinetic and nuclear-attraction launchers' `kMaxL` | [`FB_PAIR_BOUND_MAX_MOMENTUM`], [`FB_RECORD_BOUND_MAX_MOMENTUM`], [`FB_OVERLAP_MAX_MOMENTUM`], [`FB_KINETIC_MAX_MOMENTUM`], [`FB_POTENTIAL_EN_MAX_MOMENTUM`] |
//! | `wwrErrorInvalidValue` for a refusal | [`FB_LAUNCH_INVALID_ARGUMENT`] (a count, a null pointer, a short workspace) or [`FB_LAUNCH_UNSUPPORTED`] (no kernel for the class) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_integrals_abi_version`] and [`fb_integrals_backend`] against the bindings' |
//! | the other one-electron families (the dipole, the mixed overlap, the derivatives), J, K and the rest | not forked yet |

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
/// inside wwr with nothing naming the cause. Needs no device. The comparison
/// is [`fb_kernel_abi::check`], tested where no library is built.
///
/// # Errors
///
/// A mismatch, with a message naming the found and the expected values.
pub fn check_library() -> Result<(), String> {
    // SAFETY: a host function with no arguments, which calls no runtime.
    let version = unsafe { fb_integrals_abi_version() };
    // SAFETY: as above.
    let backend = unsafe { fb_integrals_backend() };
    let found = (version, backend);
    fb_kernel_abi::check(
        "integrals",
        found,
        (FB_INTEGRALS_ABI_VERSION, env!("FB_KERNELS_BACKEND")),
    )
}

// The numbers fb_kernel_abi::check names the found backend by.
const _: () = assert!(FB_BACKEND_CUDA == 1 && FB_BACKEND_HIP == 2);
// The numbers fb_kernel_abi::launch_status reads a launch status by.
const _: () = assert!(
    FB_LAUNCH_SUCCESS == 0 && FB_LAUNCH_UNSUPPORTED == -1 && FB_LAUNCH_INVALID_ARGUMENT == -2
);

// The structs' layout, which mmd_launch.cu (and potential_en_launch.cu, the
// nuclei's) static_asserts with the same numbers: both sides follow the C rule over the same fields, and these catch
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
    assert!(size_of::<fb_nuclei_view>() == 24);
    assert!(offset_of!(fb_nuclei_view, z) == 8);
    assert!(offset_of!(fb_nuclei_view, c) == 16);
};
