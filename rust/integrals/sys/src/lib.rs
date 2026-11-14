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
//! | `metric_class` (`src/integrals/eri/density_fitting/metric`) | [`fb_metric_class`], the records and pairs as field segments, as there; the safe path is [`add_metric_tasks`] over an [`AuxTable`] |
//! | `metric`'s host staging: one record per aux primitive, the pairs P <= Q per class, swapped to the lower momentum first | [`AuxTable::push_shell`] and the `metric tables` task of [`add_metric_tasks`] |
//! | `three_center_class`, `three_center_symmetrize` (`.../three_center`) | [`fb_three_center_class`], [`fb_three_center_symmetrize`]`(stream, n_ao, n_aux, input, output)`, overlapping arrays refused |
//! | `kMetricMaxMomentum`, `kThreeCenterMaxMomentum`, `kThreeCenterAuxMaxMomentum` | [`FB_METRIC_MAX_MOMENTUM`], [`FB_THREE_CENTER_MAX_MOMENTUM`], [`FB_THREE_CENTER_AUX_MAX_MOMENTUM`] |
//! | `kMetricRecordFields`, `kMetricPairFields`, `kThreeCenterBraFields`, `kThreeCenterAuxFields`, `kThreeCenterCellFields` | [`FB_METRIC_RECORD_FIELDS`], [`FB_METRIC_PAIR_FIELDS`], [`FB_THREE_CENTER_BRA_FIELDS`], [`FB_THREE_CENTER_AUX_FIELDS`], [`FB_THREE_CENTER_CELL_FIELDS`], for a safe layer to size its segments by |
//! | `wwrErrorInvalidValue` for a refusal | [`FB_LAUNCH_INVALID_ARGUMENT`] (a count, a null pointer, a short workspace) or [`FB_LAUNCH_UNSUPPORTED`] (no kernel for the class) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_integrals_abi_version`] and [`fb_integrals_backend`] against the bindings' |
//! | the other one-electron families (the dipole, the mixed overlap, the derivatives), J, K, density fitting past the metric and (mn\|P), and the rest | not forked yet |
//!
//! **The safe layer.** What a caller uses (`docs/gpu_layer.md`, "The safe
//! wrapper"), beside the generated bindings rather than in a crate of its
//! own, since this crate is already where the library's `unsafe` lives:
//!
//! - [`RecordTables`], a basis's shell table ([`ShellTable`]) and class
//!   pairs ([`ClassPairs`]) with every index the kernels follow checked
//!   when it is made, and [`add_record_tasks`], the two tasks that form
//!   every class's records from it on a stream (`fb_mmd_order`,
//!   `fb_mmd_records`, `fb_mmd_bound`, `fb_ordered_pair_bound`), returning
//!   one [`ClassRows`] per class.
//! - [`overlap_class`], [`kinetic_class`] and [`potential_en_class`], over
//!   a [`ClassRows`] and an output
//!   [`DeviceSliceMut`](fb_core_wwr::device_slice::DeviceSliceMut) whose
//!   length is asserted in every build, the status a `Status` through
//!   [`fb_kernel_abi::launch_status`].
//!
//! - [`AuxTable`], an auxiliary basis's shells and records as the
//!   density-fitting kernels read them, checked shell by shell as it is
//!   built, and [`add_metric_tasks`], the two tasks that stage it with
//!   every aux shell pair and launch each class of the Cartesian metric
//!   ([`fb_metric_class`]) into an output row. The pairs are formed here
//!   from the table and held in the tasks' own rows, so no device table a
//!   caller could fill reaches the launch.
//!
//! The three-center wrappers ([`fb_three_center_class`],
//! [`fb_three_center_symmetrize`]) have no safe call yet: the raw bindings
//! alone, for the crate that ports the module.
//!
//! A class kernel writes at AO offsets held in device memory, so
//! [`ClassRows`] is a type only [`add_record_tasks`] makes; its doc comment
//! has why a launcher may trust it.

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;
mod classes;
mod metric;
mod records;

pub use bindings::*;
pub use classes::{kinetic_class, overlap_class, potential_en_class};
pub use metric::{AuxTable, add_metric_tasks};
pub use records::{ClassPairs, ClassRows, RecordTables, ShellTable, add_record_tasks};

use fb_core_error_handling::Status;
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::gpu_status_at;

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

/// A launch status as a [`Status`] naming the wrapper and the class: a
/// launch error by its `wwr_status` name, at the safe call's caller.
#[track_caller]
fn launched(what: &str, class: &ClassRows, status: fb_launch_status) -> Status {
    let what = format!("{what} ({}, {})", class.la(), class.lb());
    fb_kernel_abi::launch_status(&what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}
