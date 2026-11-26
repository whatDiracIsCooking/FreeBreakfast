//! Density fitting's packed gather on the v0.0.4 layer: a packed batch of
//! aux functions into square matrices through [`fb_packed_gather`], over
//! `fb-core-device`'s stream and slices ([`packed_gather`]).
//!
//! The kernel stores at the `(m, n)` each column's tables hold, which it
//! reads from device memory. They come in on the host and are uploaded by
//! the call itself after every index is checked inside the matrix
//! (`docs/gpu_layer.md`, "The safe wrapper"), so no device table a caller
//! could fill reaches the launch.

use contracts::requires;
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, HostSlice, Pending, ScopedDevice, Stream, part_mut,
};
use fb_core_error_handling::Result;
use fb_core_log::{eq, fits_i32, le};

use crate::{fb_packed_gather, launched};

/// The kernel's name, which every refusal here names.
const KERNEL: &str = "fb_packed_gather";

/// Gathers the packed batch `packed`, `b` aux functions of `n_columns`
/// each, pair-fastest (element `(c, q)` at `c + n_columns q`), into
/// `slab`, `b` matrices of `n_ao × n_ao` (element `(m, n)` of function `q`
/// at `n + n_ao (m + n_ao q)`), on `stream`: the slab zeroed, `tables`
/// uploaded into the head of `tables_dev`, then one [`fb_packed_gather`]
/// launch. Every element is written, zero where no column covers it,
/// `(m, n)` and `(n, m)` the same bits; with `b = 1` it unpacks one packed
/// vector. The upload's token comes back, holding `tables` until a sync of
/// `stream` has covered it.
///
/// `tables` is each column's row `m` (`n_columns` of them), then its column
/// `n` (as many): the map's `Packing::write_tables`' first two. `weights` is
/// its third, on the device: 1 on a diagonal shell pair's column, whose
/// upper-triangle twin is then not read, and 2 on any other.
///
/// # Errors
/// The zero's or the upload's failure, or the launch's, as
/// [`fb_kernel_abi::launch_status`] maps it; after a failed launch the
/// tables are leaked with the upload's token.
///
/// # Panics
/// If a column's `m` or `n` is outside `[0, n_ao)`, naming the kernel and
/// the column, before anything is enqueued.
#[requires(
    le!(1, n_ao) && fits_i32!(n_ao),
    "fb_packed_gather: n_ao is a positive i32"
)]
#[requires(
    le!(1, n_columns) && fits_i32!(n_columns),
    "fb_packed_gather: n_columns is a positive i32"
)]
#[requires(le!(1, b) && fits_i32!(b), "fb_packed_gather: b is a positive i32")]
#[requires(
    eq!(2 * n_columns, tables.len()),
    "fb_packed_gather: tables is the rows and the columns"
)]
#[requires(
    le!(tables.len(), tables_dev.len()),
    "fb_packed_gather: tables_dev holds tables"
)]
#[requires(
    eq!(n_columns, weights.len()),
    "fb_packed_gather: weights is a column each"
)]
#[requires(
    eq!(Some(packed.len()), b.checked_mul(n_columns)),
    "fb_packed_gather: packed is b × n_columns doubles"
)]
#[requires(
    eq!(Some(slab.len()), n_ao.checked_mul(n_ao).and_then(|s| s.checked_mul(b))),
    "fb_packed_gather: slab is b × n_ao² doubles"
)]
#[track_caller]
pub fn packed_gather<'s, H: HostSlice<i32>>(
    stream: &'s Stream,
    (n_ao, n_columns, b): (usize, usize, usize),
    tables: H,
    tables_dev: &mut impl DeviceSliceMut<i32>,
    weights: &impl DeviceSlice<f64>,
    packed: &impl DeviceSlice<f64>,
    slab: &mut impl DeviceSliceMut<f64>,
) -> Result<Pending<'s, H>> {
    {
        // SAFETY: `len()` elements of host memory reached only through
        // `tables` (HostSlice's promise), borrowed shared for this block.
        let t = unsafe { std::slice::from_raw_parts(tables.as_ptr(), tables.len()) };
        let (rows, cols) = t.split_at(n_columns);
        check_indices(KERNEL, n_ao, rows, cols);
    }
    stream.zero(slab)?;
    let len = tables.len();
    let uploaded = stream.upload(&mut part_mut(tables_dev, 0, len), tables)?;
    let rows = tables_dev.as_ptr();
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: rows and cols are the n_columns integers each the upload just
    // before on this stream filled from host memory its token holds
    // unwritten until a sync covers the copy, every one checked inside
    // [0, n_ao) (check_indices), so every store at (m, n) and (n, m) of a
    // matrix is inside its n_ao² doubles and every matrix inside slab's b
    // of them, as every read of packed is inside its b × n_columns and of
    // weights inside its n_columns (the contracts, each count a positive
    // i32, so the casts are exact); slab and tables_dev have no other user
    // until the enqueued work finishes (DeviceSliceMut's promise), so apart
    // from packed and weights; the call only enqueues, after the zero on the
    // same stream.
    let status = unsafe {
        fb_packed_gather(
            stream.raw(),
            n_ao as i32,
            n_columns as i32,
            b as i32,
            rows,
            rows.wrapping_add(n_columns),
            weights.as_ptr(),
            packed.as_ptr(),
            slab.as_mut_ptr(),
        )
    };
    launched(KERNEL, &[], status)?;
    Ok(uploaded)
}

/// Every `m` and `n` inside `[0, nao)`: what the kernel's stores rest on.
/// `what` names the caller in the message.
///
/// # Panics
/// At the first column outside, naming it and its `(m, n)`.
#[track_caller]
fn check_indices(what: &str, nao: usize, rows: &[i32], cols: &[i32]) {
    let inside = |i: i32| usize::try_from(i).is_ok_and(|i| i < nao);
    for (c, (&m, &n)) in rows.iter().zip(cols).enumerate() {
        assert!(
            inside(m) && inside(n),
            "{what}: column {c} stands at ({m}, {n}), outside a {nao} × {nao} matrix"
        );
    }
}
