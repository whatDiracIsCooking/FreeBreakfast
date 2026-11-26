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
//! | `mmd_order(stream, n_rec, src, key, key_alt, perm, perm_alt)`, each class sorted into descending ζ | nothing: the records stay in enumeration order, which no reader here needs changed |
//! | `mmd_records(stream, n_rec, src, perm, out)`, `mmd_bound` | [`fb_mmd_records`]`(stream, n_rec, src, out)`, record r the host's record r; [`fb_mmd_bound`]; the safe call is [`mmd_records`] |
//! | `ordered_pair_bound` | [`fb_ordered_pair_bound`]; the safe call is [`record_bound`] |
//! | `overlap_class`, `kinetic_class` | [`fb_overlap_class`], [`fb_kinetic_class`], with no safe call: the v0.0.4 layer forms S and T on the host |
//! | `NucleiView` (`src/molecule`) | [`fb_nuclei_view`], `#[repr(C)]`, by pointer |
//! | `potential_en_class(stream, cls, nuclei, n_ao, v)` | [`fb_potential_en_class`]`(stream, cls, nuclei, n_ao, v)`, the nuclei by pointer, with no safe call: the v0.0.4 layer forms V on the host |
//! | `kPairBoundMaxMomentum`, `kRecordBoundMaxMomentum`, the overlap, kinetic and nuclear-attraction launchers' `kMaxL` | [`FB_PAIR_BOUND_MAX_MOMENTUM`], [`FB_RECORD_BOUND_MAX_MOMENTUM`], [`FB_OVERLAP_MAX_MOMENTUM`], [`FB_KINETIC_MAX_MOMENTUM`], [`FB_POTENTIAL_EN_MAX_MOMENTUM`] |
//! | `metric_class` (`src/integrals/eri/density_fitting/metric`) | [`fb_metric_class`], the records and pairs as field segments, as there, with no safe call: the v0.0.4 layer forms the metric on the host |
//! | `three_center_class(stream, la, lb, lp, n_bra_rec, bra, …)`, `three_center_symmetrize` (`.../three_center`) | nothing: the square tensor is not built (#1721), and its kernels and launchers are not carried (#1725) |
//! | nothing: the C++ (mn\|P) is square | [`fb_three_center_packed_class`], rust/'s own launcher over FelizJueves's packed spherical-aux `speri3<la, lb, lP>` units, each cell's block at its destination column, the aux index already spherical, and [`FB_THREE_CENTER_PACKED_CELL_FIELDS`], with no safe call |
//! | nothing: the C++ (mn\|P) is Cartesian in the bra | [`fb_three_center_spherical_class`], the same launcher over FelizJueves's fully spherical `ssperi3<la, lb, lP>` (#745), each cell's (μν\|Q) block spherical in all three indices; the safe call is [`three_center_spherical_class`] |
//! | `kMetricMaxMomentum`, `kThreeCenterMaxMomentum`, `kThreeCenterAuxMaxMomentum` | [`FB_METRIC_MAX_MOMENTUM`], [`FB_THREE_CENTER_MAX_MOMENTUM`], [`FB_THREE_CENTER_AUX_MAX_MOMENTUM`] |
//! | `kMetricRecordFields`, `kMetricPairFields`, `kThreeCenterAuxFields` | [`FB_METRIC_RECORD_FIELDS`], [`FB_METRIC_PAIR_FIELDS`], [`FB_THREE_CENTER_AUX_FIELDS`], for a safe layer to size its segments by; `kThreeCenterCellFields`, the square cell's, has nothing (#1725) |
//! | `kThreeCenterBraFields`, the packed bra records' segment count | nothing: the bra records cross as a class view, a field per pointer |
//! | `wwrErrorInvalidValue` for a refusal | [`FB_LAUNCH_INVALID_ARGUMENT`] (a count, a null pointer, a short workspace) or [`FB_LAUNCH_UNSUPPORTED`] (no kernel for the class) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_integrals_abi_version`] and [`fb_integrals_backend`] against the bindings' |
//! | nothing: the C++ tree has no conventional mode | [`fb_conventional_eri_class`], [`fb_conventional_exchange`], [`fb_conventional_pack`] and [`fb_conventional_unpack`], rust/'s own launchers over FelizJueves's `eri/full_eri` units (`eri<la, lb, lc, ld>`, `eri_kperm`), and [`FB_CONVENTIONAL_MAX_MOMENTUM`], [`FB_CONVENTIONAL_MAX_AO`], [`FB_CONVENTIONAL_PAIR_FIELDS`], [`FB_CONVENTIONAL_TASK_FIELDS`], with no safe call |
//! | nothing: the C++ tree has no conventional mode | [`fb_dense_eri_class`], rust/'s own launcher over FelizJueves's dense `eri/full_eri` units (`eri_dj<la, lb, lc, ld>`, `eri_dk<la, lb, lc, ld>`), [`FB_DENSE_LAYOUT_J`], [`FB_DENSE_LAYOUT_K`]; and [`fb_dense_row_dot`], a row-major matrix by a vector in an order fixed by its width, with no safe call |
//! | nothing: the C++ B is square and resident | [`fb_packed_pack`] and [`fb_packed_gather`], rust/'s own pack of a density into density fitting's packed shell-pair map and gather of packed columns back into the square; the gather's safe call is [`packed_gather`], the pack has none |
//! | the other one-electron families (the dipole, the mixed overlap, the derivatives), J, K, density fitting past the metric and (mn\|P), and the rest | not forked yet |
//!
//! **The safe layer.** What a caller uses (`docs/gpu_layer.md`, "The safe
//! wrapper"), beside the generated bindings rather than in a crate of its
//! own, since this crate is already where the library's `unsafe` lives.
//! Each call is the v0.0.4 layer's, over `fb-core-device`'s stream and
//! slices, every length a contract, and every index a kernel follows
//! handed in on the host, checked inside what it indexes and uploaded by
//! the call itself before the launch, so no device table a caller could
//! fill reaches a kernel:
//!
//! - [`mmd_records`], one class's primitive-pair records through
//!   [`fb_mmd_records`], its shell and pair tables in one table.
//! - [`record_bound`], each record's dressed Schwarz bound through
//!   [`fb_ordered_pair_bound`] from the records [`mmd_records`] wrote,
//!   which the eri step's screen sums per shell pair.
//! - [`three_center_spherical_class`], one class of the fully spherical
//!   packed (μν\|Q) through [`fb_three_center_spherical_class`]
//!   (`ssperi3<la, lb, lP>`), every cell checked inside the slices it
//!   indexes.
//! - [`packed_gather`], a packed batch gathered into square matrices
//!   through [`fb_packed_gather`], each column's row and column checked
//!   inside the matrix.
//!
//! The launchers with no safe call are the old layer's, whose task
//! builders were deleted with it (#1874); each is tested raw, through its
//! bindings, under `tests/`.

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;
mod mmd_records;
mod packed_gather;
mod three_center_spherical;

pub use bindings::*;
pub use mmd_records::{RECORD_F64_FIELDS, RECORD_I32_FIELDS, mmd_records, record_bound};
pub use packed_gather::packed_gather;
pub use three_center_spherical::three_center_spherical_class;

use fb_core_error_handling::Status;
use fb_kernel_abi::Backend;

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
    let built_for = const { Backend::named(env!("FB_KERNELS_BACKEND")) };
    fb_kernel_abi::check(
        "integrals",
        (version, backend),
        (FB_INTEGRALS_ABI_VERSION, built_for),
    )
}

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

/// A launch status as a [`Status`] naming the wrapper and the class by its
/// momenta, `fb_metric_class (2, 3)`, or the wrapper alone when `momenta`
/// is empty, as [`fb_core_device::launch_status`] maps it, at the safe
/// call's caller.
#[track_caller]
fn launched(what: &str, momenta: &[i32], status: fb_launch_status) -> Status {
    let what = if momenta.is_empty() {
        what.to_owned()
    } else {
        let momenta: Vec<String> = momenta.iter().map(i32::to_string).collect();
        format!("{what} ({})", momenta.join(", "))
    };
    fb_core_device::launch_status(&what, status)
}
