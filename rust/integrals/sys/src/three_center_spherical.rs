//! The fully spherical packed three-center integrals on the v0.0.4 layer:
//! one class `(la, lb, lP)` of `(μν|Q)`, all three indices real solid
//! harmonics, through FelizJueves's `ssperi3<la, lb, lP>` (#745), over
//! `fb-core-device`'s stream and slices ([`three_center_spherical_class`]).
//!
//! The kernel follows indices it reads from device memory: each cell's
//! record ranges and its destination. Those come in on the host and are
//! uploaded by the call itself, after each is checked against the slices
//! they index (`docs/gpu_layer.md`, "The safe wrapper": where the safe layer
//! forms the indices itself, the staging and the launch are the binding
//! crate's together), so no device table a caller could fill reaches the
//! launch.

use contracts::requires;
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, HostSlice, Pending, ScopedDevice, Stream, part_mut,
};
use fb_core_error_handling::Result;
use fb_core_log::{eq, fits_i32, le};

use crate::{
    FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_AUX_MAX_MOMENTUM, FB_THREE_CENTER_MAX_MOMENTUM,
    FB_THREE_CENTER_PACKED_CELL_FIELDS, fb_primitive_pair_class_view,
    fb_three_center_spherical_class, launched,
};

/// The record fields the kernel reads, the first eleven `fb_mmd_records`
/// writes: zeta, P, PA, PB and s.
const BRA_FIELDS: usize = 11;
/// The fields of one aux record.
const AUX_FIELDS: usize = FB_THREE_CENTER_AUX_FIELDS as usize;
/// The fields of one cell.
const CELL_FIELDS: usize = FB_THREE_CENTER_PACKED_CELL_FIELDS as usize;
/// The kernel's name, which every refusal here names.
const KERNEL: &str = "fb_three_center_spherical_class";

/// Class `(la, lb, lp)` of the fully spherical packed `(μν|Q)` into `eri3`
/// on `stream`: `cells` uploaded into the head of `cells_dev`, then one
/// [`fb_three_center_spherical_class`] launch. The upload's token comes
/// back, holding `cells` until a sync of `stream` has covered it.
///
/// - `records` is the class's records, field-major, field `k` of record `r`
///   at `k n_rec + r`, as `fb_mmd_records` writes them; the first eleven
///   fields (zeta, P, PA, PB, s) are read.
/// - `aux` is the aux records, field-major, [`FB_THREE_CENTER_AUX_FIELDS`]
///   fields of `n_aux_rec`: centre, exponent and weight
///   `c (π/α)^{3/2}`.
/// - `cells` is [`FB_THREE_CENTER_PACKED_CELL_FIELDS`] fields of `n_cell`,
///   field-major: each cell's bra record range among `records`, its aux
///   record range among `aux`, its destination `dst` and its diagonal flag.
///   Element `(μ, ν, q)` of a cell is stored at
///   `dst + n_aux (μ (2 lb + 1) + ν) + q`.
/// - `eri3` is the batch, `n_col` spherical bra columns of `n_aux` aux
///   functions, aux fastest.
///
/// Every element of a cell's block is a plain store, the block zeroed by
/// its own cell first, so the batch needs no zeroing and cells whose blocks
/// do not overlap need no ordering.
///
/// # Errors
/// The upload's failure, or the launch's, as
/// [`fb_kernel_abi::launch_status`] maps it; after a failed launch the
/// cells are leaked with the upload's token.
///
/// # Panics
/// If a cell's bra record range is not inside `[0, n_rec]`, its aux record
/// range not inside `[0, n_aux_rec]`, or its block not inside `eri3`,
/// naming the kernel and the cell, before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the kernel's arguments")]
#[requires(
    le!(1, n_rec) && fits_i32!(n_rec),
    "fb_three_center_spherical_class: n_rec is a positive i32"
)]
#[requires(
    le!(BRA_FIELDS * n_rec, records.len()),
    "fb_three_center_spherical_class: records holds 11 fields of n_rec"
)]
#[requires(
    le!(1, n_aux_rec) && fits_i32!(n_aux_rec),
    "fb_three_center_spherical_class: n_aux_rec is a positive i32"
)]
#[requires(
    le!(AUX_FIELDS * n_aux_rec, aux.len()),
    "fb_three_center_spherical_class: aux holds 5 fields of n_aux_rec"
)]
#[requires(
    le!(1, n_cell) && fits_i32!(n_cell),
    "fb_three_center_spherical_class: n_cell is a positive i32"
)]
#[requires(
    eq!(CELL_FIELDS * n_cell, cells.len()),
    "fb_three_center_spherical_class: cells is 6 fields of n_cell"
)]
#[requires(
    le!(cells.len(), cells_dev.len()),
    "fb_three_center_spherical_class: cells_dev holds cells"
)]
#[requires(
    le!(1, n_col) && le!(1, n_aux) && fits_i32!(n_col.checked_mul(n_aux)),
    "fb_three_center_spherical_class: n_col × n_aux is a positive i32"
)]
#[requires(
    le!(n_col * n_aux, eri3.len()),
    "fb_three_center_spherical_class: eri3 holds n_col × n_aux"
)]
// RELEASE: the launcher refuses a class outside its table as
// FB_LAUNCH_UNSUPPORTED, enqueueing nothing.
#[debug_requires(
    0 <= la && la <= lb && lb <= FB_THREE_CENTER_MAX_MOMENTUM,
    "the bra class has a kernel"
)]
// RELEASE: as above.
#[debug_requires(
    (0..=FB_THREE_CENTER_AUX_MAX_MOMENTUM).contains(&lp),
    "the aux momentum has a kernel"
)]
#[track_caller]
pub fn three_center_spherical_class<'s, H: HostSlice<i32>>(
    stream: &'s Stream,
    (la, lb, lp): (i32, i32, i32),
    n_rec: usize,
    records: &impl DeviceSlice<f64>,
    n_aux_rec: usize,
    aux: &impl DeviceSlice<f64>,
    n_cell: usize,
    cells: H,
    cells_dev: &mut impl DeviceSliceMut<i32>,
    (n_col, n_aux): (usize, usize),
    eri3: &mut impl DeviceSliceMut<f64>,
) -> Result<Pending<'s, H>> {
    check_cells(
        &cells,
        n_cell,
        (n_rec, n_aux_rec),
        block_span((la, lb, lp), n_aux),
        n_col * n_aux,
    );
    let len = cells.len();
    let uploaded = stream.upload(&mut part_mut(cells_dev, 0, len), cells)?;
    let rec = records.as_ptr();
    let field = |k: usize| rec.wrapping_add(k * n_rec);
    let null = std::ptr::null();
    let view = fb_primitive_pair_class_view {
        la,
        lb,
        n_rec: n_rec as i32,
        zeta: field(0),
        p_x: field(1),
        p_y: field(2),
        p_z: field(3),
        pa_x: field(4),
        pa_y: field(5),
        pa_z: field(6),
        pb_x: field(7),
        pb_y: field(8),
        pb_z: field(9),
        s: field(10),
        ea: null,
        eb: null,
        bound: null,
        bound_ab: null,
        bound_ba: null,
        off_a: std::ptr::null(),
        off_b: std::ptr::null(),
        rank_a: std::ptr::null(),
        rank_b: std::ptr::null(),
    };
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: the view's eleven fields are runs of n_rec doubles inside
    // records, the aux records' five of n_aux_rec inside aux, and eri3 holds
    // n_col n_aux doubles (the contracts, each count a positive i32, so the
    // casts are exact); every cell, checked here and then uploaded just
    // before on this stream from host memory the upload's token holds
    // unwritten until a sync covers the copy, has its record ranges inside
    // those runs and its whole block inside eri3 (check_cells), the only
    // indices the kernel follows; eri3
    // and cells_dev have no other user until the enqueued work finishes
    // (DeviceSliceMut's promise); the view is a host value read before the
    // call returns, which only enqueues.
    let status = unsafe {
        fb_three_center_spherical_class(
            stream.raw(),
            &view,
            lp,
            n_aux_rec as i32,
            aux.as_ptr(),
            n_cell as i32,
            cells_dev.as_ptr(),
            n_col as i32,
            n_aux as i32,
            eri3.as_mut_ptr(),
        )
    };
    launched(KERNEL, &[la, lb, lp], status)?;
    Ok(uploaded)
}

/// The elements a cell's block spans past its `dst`, one more than its last
/// element's offset: `n_aux ((2 la + 1)(2 lb + 1) - 1) + 2 lp + 1`, widened
/// so it cannot wrap. A momentum outside the table gives what it gives: the
/// launcher refuses that class before reading a cell.
fn block_span((la, lb, lp): (i32, i32, i32), n_aux: usize) -> i64 {
    let n = |l: i32| 2 * i64::from(l) + 1;
    n_aux as i64 * (n(la) * n(lb) - 1) + n(lp)
}

/// Asserts that each of the `n_cell` cells in `cells` (field-major) has its
/// bra record range inside `[0, n_rec]`, its aux record range inside
/// `[0, n_aux_rec]`, and its block, `span` elements from its `dst`, inside
/// the `len` elements of the batch: what the kernel indexes by.
#[track_caller]
fn check_cells(
    cells: &impl HostSlice<i32>,
    n_cell: usize,
    (n_rec, n_aux_rec): (usize, usize),
    span: i64,
    len: usize,
) {
    // SAFETY: `len()` elements of host memory reached only through `cells`
    // (HostSlice's promise), borrowed shared here.
    let cells = unsafe { std::slice::from_raw_parts(cells.as_ptr(), cells.len()) };
    let field = |f: usize, k: usize| i64::from(cells[f * n_cell + k]);
    let inside = |lo: i64, hi: i64, end: usize| 0 <= lo && lo <= hi && hi <= end as i64;
    for k in 0..n_cell {
        let (b0, b1, p0, p1, dst) = (
            field(0, k),
            field(1, k),
            field(2, k),
            field(3, k),
            field(4, k),
        );
        assert!(
            inside(b0, b1, n_rec),
            "{KERNEL}: cell {k}'s bra records [{b0}, {b1}) are not inside [0, {n_rec})"
        );
        assert!(
            inside(p0, p1, n_aux_rec),
            "{KERNEL}: cell {k}'s aux records [{p0}, {p1}) are not inside [0, {n_aux_rec})"
        );
        assert!(
            0 <= dst && dst + span <= len as i64,
            "{KERNEL}: cell {k}'s block [{dst}, {}) is not inside eri3's {len}",
            dst + span
        );
    }
}
