//! The primitive-pair records on the v0.0.4 layer: one class's records
//! through [`fb_mmd_records`] over `fb-core-device`'s stream and slices
//! ([`mmd_records`]), and each record's dressed Schwarz bound through
//! [`fb_ordered_pair_bound`] ([`record_bound`]).
//!
//! The kernel follows indices it reads from device memory: each shell's
//! primitive range, each pair's two shells and each pair's record range.
//! Those come in on the host, as one table, and are uploaded by the call
//! itself after each is checked against what it indexes
//! (`docs/gpu_layer.md`, "The safe wrapper"), so no device table a caller
//! could fill reaches the launch. The old layer's [`add_record_tasks`]
//! forms the same records from the same tables, checked by
//! [`RecordTables::new`].
//!
//! [`add_record_tasks`]: crate::add_record_tasks
//! [`RecordTables::new`]: crate::RecordTables::new

use contracts::requires;
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, HostSlice, Pending, ScopedDevice, Stream, part_mut,
};
use fb_core_error_handling::{Result, Status};
use fb_core_log::{eq, fits_i32, le};

use crate::{
    FB_RECORD_BOUND_MAX_MOMENTUM, fb_mmd_records, fb_ordered_pair_bound,
    fb_primitive_pair_class_view, fb_primitive_pair_records, fb_primitive_pair_source, launched,
};

/// The double fields of one record [`mmd_records`] writes, in their order:
/// zeta, P (x, y, z), P - A, P - B, s, alpha and beta.
pub const RECORD_F64_FIELDS: usize = 13;
/// The integer fields of one record, in their order: the bra and ket
/// shells' first Cartesian AO, and their ranks.
pub const RECORD_I32_FIELDS: usize = 4;
/// The kernel's name, which every refusal here names.
const KERNEL: &str = "fb_mmd_records";

/// One class's primitive-pair records on `stream`: `tables` uploaded into
/// the head of `tables_dev`, then one [`fb_mmd_records`] launch. The
/// upload's token comes back, holding `tables` until a sync of `stream` has
/// covered it.
///
/// - `shells` is the basis's shell table's doubles, field-major: every
///   primitive's exponent, then every coefficient (`n_prim` each, shell by
///   shell), then each shell's centre x, y and z (`n_shell` each).
/// - `tables` is its integers and the class's pairs, one after the other:
///   `prim_begin` (`n_shell + 1`, each shell's first primitive), `ao_offset`
///   and `rank` (`n_shell` each, each shell's first Cartesian AO and its
///   rank among the shells of its momentum), `rec_begin` (`n_pairs + 1`,
///   each pair's first record), and the pairs' `bra` and `ket` shells
///   (`n_pairs` each), `3 n_shell + 3 n_pairs + 2` in all.
/// - `f64_fields` and `i32_fields` receive the records, field `k` of record
///   `r` at `k n_rec + r`: [`RECORD_F64_FIELDS`] doubles and
///   [`RECORD_I32_FIELDS`] integers of `n_rec`, in enumeration order (shell
///   pair, then bra primitive, then ket primitive). The first eleven double
///   fields are what [`three_center_spherical_class`] reads.
///
/// `ao_offset` and `rank` are copied into the records and followed by no
/// kernel here, so they are not checked: a launch on this layer that
/// follows a record's AO offset checks it itself.
///
/// # Errors
/// The upload's failure, or the launch's, as
/// [`fb_kernel_abi::launch_status`] maps it; after a failed launch the
/// tables are leaked with the upload's token.
///
/// # Panics
/// If `prim_begin` does not start at 0, decreases, or does not end at
/// `n_prim`; a pair names a shell outside `[0, n_shell)`; or `rec_begin`
/// does not start at 0, does not step by each pair's primitive product, or
/// does not end at `n_rec`, naming the kernel and the entry, before anything
/// is enqueued.
///
/// [`three_center_spherical_class`]: crate::three_center_spherical_class
#[allow(clippy::too_many_arguments, reason = "the kernel's arguments")]
#[requires(
    le!(1, n_shell) && fits_i32!(n_shell),
    "fb_mmd_records: n_shell is a positive i32"
)]
#[requires(
    le!(1, n_prim) && fits_i32!(n_prim),
    "fb_mmd_records: n_prim is a positive i32"
)]
#[requires(
    le!(2 * n_prim + 3 * n_shell, shells.len()),
    "fb_mmd_records: shells holds the exponents, coefficients and centres"
)]
#[requires(
    le!(1, n_pairs) && fits_i32!(n_pairs),
    "fb_mmd_records: n_pairs is a positive i32"
)]
#[requires(
    le!(1, n_rec) && fits_i32!(n_rec),
    "fb_mmd_records: n_rec is a positive i32"
)]
#[requires(
    eq!(3 * n_shell + 3 * n_pairs + 2, tables.len()),
    "fb_mmd_records: tables is the shell and pair tables"
)]
#[requires(
    le!(tables.len(), tables_dev.len()),
    "fb_mmd_records: tables_dev holds tables"
)]
#[requires(
    le!(RECORD_F64_FIELDS * n_rec, f64_fields.len()),
    "fb_mmd_records: f64_fields holds 13 fields of n_rec"
)]
#[requires(
    le!(RECORD_I32_FIELDS * n_rec, i32_fields.len()),
    "fb_mmd_records: i32_fields holds 4 fields of n_rec"
)]
#[track_caller]
pub fn mmd_records<'s, H: HostSlice<i32>>(
    stream: &'s Stream,
    (n_shell, n_prim): (usize, usize),
    shells: &impl DeviceSlice<f64>,
    (n_pairs, n_rec): (usize, usize),
    tables: H,
    tables_dev: &mut impl DeviceSliceMut<i32>,
    f64_fields: &mut impl DeviceSliceMut<f64>,
    i32_fields: &mut impl DeviceSliceMut<i32>,
) -> Result<Pending<'s, H>> {
    check_tables(&tables, (n_shell, n_prim), (n_pairs, n_rec));
    let len = tables.len();
    let uploaded = stream.upload(&mut part_mut(tables_dev, 0, len), tables)?;
    // Where each table starts in the upload, as the doc comment lists them.
    let (ao_at, rank_at, rec_at) = (n_shell + 1, 2 * n_shell + 1, 3 * n_shell + 1);
    let (bra_at, ket_at) = (rec_at + n_pairs + 1, rec_at + 2 * n_pairs + 1);
    let int = tables_dev.as_ptr();
    let shell = shells.as_ptr();
    let src = fb_primitive_pair_source {
        n_pairs: n_pairs as i32,
        bra: int.wrapping_add(bra_at).cast(),
        ket: int.wrapping_add(ket_at).cast(),
        rec_begin: int.wrapping_add(rec_at),
        prim_begin: int,
        exponent: shell,
        coefficient: shell.wrapping_add(n_prim),
        c_x: shell.wrapping_add(2 * n_prim),
        c_y: shell.wrapping_add(2 * n_prim + n_shell),
        c_z: shell.wrapping_add(2 * n_prim + 2 * n_shell),
        ao_offset: int.wrapping_add(ao_at),
        rank: int.wrapping_add(rank_at),
    };
    let f = f64_fields.as_mut_ptr();
    let fd = |k: usize| f.wrapping_add(k * n_rec);
    let i = i32_fields.as_mut_ptr();
    let fi = |k: usize| i.wrapping_add(k * n_rec);
    let out = fb_primitive_pair_records {
        zeta: fd(0),
        p_x: fd(1),
        p_y: fd(2),
        p_z: fd(3),
        pa_x: fd(4),
        pa_y: fd(5),
        pa_z: fd(6),
        pb_x: fd(7),
        pb_y: fd(8),
        pb_z: fd(9),
        s: fd(10),
        ea: fd(11),
        eb: fd(12),
        off_a: fi(0),
        off_b: fi(1),
        rank_a: fi(2),
        rank_b: fi(3),
    };
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: the source's double arrays are runs of n_prim and n_shell
    // doubles inside shells, and its integer arrays runs of tables_dev the
    // upload just before on this stream filled from host memory its token
    // holds unwritten until a sync covers the copy; each record field is a
    // run of n_rec inside f64_fields or i32_fields (the contracts, each
    // count a positive i32, so the casts are exact). The kernel follows
    // only prim_begin, bra, ket and rec_begin, checked here: every pair's
    // shells inside [0, n_shell), every shell's primitives inside
    // [0, n_prim), and the record ranges a prefix stepping by each pair's
    // primitive product to n_rec, so every record's pair search, primitive
    // and shell index stays inside them (check_tables). f64_fields,
    // i32_fields and tables_dev have no other user until the enqueued work
    // finishes (DeviceSliceMut's promise); both structs are host values read
    // before the call returns, which only enqueues.
    let status = unsafe { fb_mmd_records(stream.raw(), n_rec as i32, &src, &out) };
    launched(KERNEL, &[], status)?;
    Ok(uploaded)
}

/// The record fields [`fb_ordered_pair_bound`] reads, the first eleven
/// [`mmd_records`] writes: zeta, P, P - A, P - B and s (P itself unread).
const BOUND_FIELDS: usize = 11;

/// Each of class `(la, lb)`'s `n_rec` records' dressed Schwarz bound into
/// `bound` on `stream`, one [`fb_ordered_pair_bound`] launch:
/// `max_{i ∈ a} Σ_{k ∈ b} Σ_p |s E^{a_i b_k}_p| I_p(ζ)`, the max over the
/// bra shell's Cartesians, `I_p` the Hermite Gaussian's Coulomb self-norm.
/// By the triangle inequality the sum of a shell pair's record bounds is
/// at least `sqrt((mn|mn))` for every Cartesian `m` of its bra and `n` of
/// its ket. `records` is the class's records as [`mmd_records`] writes
/// them, field `k` of record `r` at `k n_rec + r`.
///
/// # Errors
/// The launch's failure, as [`fb_kernel_abi::launch_status`] maps it.
#[requires(
    le!(1, n_rec) && fits_i32!(n_rec),
    "fb_ordered_pair_bound: n_rec is a positive i32"
)]
#[requires(
    le!(BOUND_FIELDS * n_rec, records.len()),
    "fb_ordered_pair_bound: records holds 11 fields of n_rec"
)]
#[requires(
    le!(n_rec, bound.len()),
    "fb_ordered_pair_bound: bound holds n_rec"
)]
// RELEASE: the launcher refuses a class outside its table as
// FB_LAUNCH_UNSUPPORTED, enqueueing nothing.
#[debug_requires(
    0 <= la && la <= lb && lb <= FB_RECORD_BOUND_MAX_MOMENTUM,
    "the class has a kernel"
)]
#[track_caller]
pub fn record_bound(
    stream: &Stream,
    (la, lb): (i32, i32),
    n_rec: usize,
    records: &impl DeviceSlice<f64>,
    bound: &mut impl DeviceSliceMut<f64>,
) -> Status {
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
    // records, of which the kernel reads zeta, P - A, P - B and s, and
    // bound holds n_rec doubles (the contracts, n_rec a positive i32, so
    // the cast is exact); the kernel is one thread per record and follows
    // no index; bound has no other user until the enqueued work finishes
    // (DeviceSliceMut's promise); the view is a host value read before the
    // call returns, which only enqueues.
    let status = unsafe { fb_ordered_pair_bound(stream.raw(), &view, bound.as_mut_ptr()) };
    launched("fb_ordered_pair_bound", &[la, lb], status)
}

/// Asserts that `tables`, laid out as [`mmd_records`] takes them, holds a
/// `prim_begin` that is a prefix of `n_prim` primitives, every pair's
/// shells inside `[0, n_shell)`, and a `rec_begin` from 0 stepping by each
/// pair's primitive product to `n_rec`: what the kernel indexes by.
#[track_caller]
fn check_tables(
    tables: &impl HostSlice<i32>,
    (n_shell, n_prim): (usize, usize),
    (n_pairs, n_rec): (usize, usize),
) {
    // SAFETY: `len()` elements of host memory reached only through `tables`
    // (HostSlice's promise), borrowed shared here.
    let t = unsafe { std::slice::from_raw_parts(tables.as_ptr(), tables.len()) };
    let (prim_begin, rest) = t.split_at(n_shell + 1);
    let (rec_begin, pairs) = rest[2 * n_shell..].split_at(n_pairs + 1);
    let (bra, ket) = pairs.split_at(n_pairs);
    assert!(
        prim_begin[0] == 0 && prim_begin.is_sorted() && prim_begin[n_shell] as usize == n_prim,
        "{KERNEL}: prim_begin is not a prefix of the {n_prim} primitives"
    );
    // Non-negative: prim_begin is a prefix, checked above.
    let prims = |shell: usize| i64::from(prim_begin[shell + 1] - prim_begin[shell]);
    assert!(
        rec_begin[0] == 0,
        "{KERNEL}: rec_begin starts at {}, not 0",
        rec_begin[0]
    );
    let inside = |s: i32| usize::try_from(s).is_ok_and(|s| s < n_shell);
    for p in 0..n_pairs {
        let (a, b) = (bra[p], ket[p]);
        assert!(
            inside(a) && inside(b),
            "{KERNEL}: pair {p} names shells {a} and {b} of {n_shell}"
        );
        let step = i64::from(rec_begin[p + 1]) - i64::from(rec_begin[p]);
        assert!(
            step == prims(a as usize) * prims(b as usize),
            "{KERNEL}: pair {p}'s records [{}, {}) are not its primitive product",
            rec_begin[p],
            rec_begin[p + 1]
        );
    }
    assert!(
        rec_begin[n_pairs] as usize == n_rec,
        "{KERNEL}: rec_begin ends at {}, not n_rec = {n_rec}",
        rec_begin[n_pairs]
    );
}
