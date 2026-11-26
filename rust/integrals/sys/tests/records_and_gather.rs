//! The v0.0.4 layer's records and gather calls, `mmd_records` and
//! `packed_gather`, refusing what their kernels' memory safety rests on
//! (tier 4): a slice one element short, and a table index outside what it
//! indexes, each refused before anything is enqueued, naming the kernel.
//! Every device buffer a call could write is filled first, and still holds
//! the fill after the refusals. Each call's results against the old
//! layer's are `fb-integrals-mmd`'s `tests/stream_records.rs` and
//! `fb-integrals-eri-density-fitting-pair-map`'s `tests/device.rs`.
//!
//! The inputs are the smallest that reach every check: two shells, s of
//! two primitives and p of one, and their one pair, `(0, 1)`, two records;
//! and a 3 × 3 matrix packed into two columns.

use std::panic::{AssertUnwindSafe, catch_unwind};

use fb_core_budget::{Backing, Budget, Caps, DeviceBuf, Ledger, PinnedBuf, Plain};
use fb_core_device::{DeviceSlice, Stream};
use fb_integrals_sys::{RECORD_F64_FIELDS, RECORD_I32_FIELDS, mmd_records, packed_gather};

/// The byte every writable buffer is filled with.
const SENTINEL: u8 = 0xA5;

/// The panic message of `f`, which must panic.
fn refusal(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("the call is refused");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

/// A real ledger on device 0, for the buffers.
fn ledger() -> Ledger {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    Ledger::new(caps, Backing::Real, vec![0], false)
}

/// `data` in pinned memory, the host end of a call.
fn pinned<T: Plain>(budget: &Budget<'_>, data: &[T]) -> PinnedBuf<T> {
    let mut p = budget.pinned::<T>(0, data.len());
    p.copy_from_slice(data);
    p
}

/// `data` uploaded into a device buffer.
fn device<T: Plain>(budget: &Budget<'_>, stream: &Stream, data: &[T]) -> DeviceBuf<T> {
    let mut d = budget.device::<T>(0, data.len());
    drop(
        stream
            .upload(&mut d, pinned(budget, data))
            .unwrap()
            .sync()
            .unwrap(),
    );
    d
}

/// A device buffer of `len` elements, every byte [`SENTINEL`].
fn filled<T: Plain>(budget: &Budget<'_>, stream: &Stream, len: usize) -> DeviceBuf<T> {
    let mut d = budget.device::<T>(0, len);
    stream.fill(&mut d, SENTINEL).unwrap();
    d
}

/// `d` downloaded.
fn read<T: Plain>(budget: &Budget<'_>, stream: &Stream, d: &DeviceBuf<T>) -> Vec<T> {
    let host = budget.pinned::<T>(0, d.len());
    stream.download(host, d).unwrap().sync().unwrap().to_vec()
}

/// Every message names `kernel`; every buffer still holds its fill.
fn nothing_enqueued(
    kernel: &str,
    messages: &[String],
    doubles: &[(&str, Vec<f64>)],
    ints: &[(&str, Vec<i32>)],
) {
    for m in messages {
        assert!(
            m.contains(kernel),
            "a refusal that does not name {kernel}: {m}"
        );
    }
    let (f, i) = (
        f64::from_ne_bytes([SENTINEL; 8]).to_bits(),
        i32::from_ne_bytes([SENTINEL; 4]),
    );
    for (what, v) in doubles {
        assert!(v.iter().all(|x| x.to_bits() == f), "{what} was written");
    }
    for (what, v) in ints {
        assert!(v.iter().all(|&x| x == i), "{what} was written");
    }
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_records_refuse_a_short_slice_or_a_stray_index_before_anything_is_enqueued() {
    let ledger = ledger();
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let (n_shell, n_prim, n_pairs, n_rec) = (2, 3, 1, 2);
    // Exponents, coefficients, then the centres' x, y and z.
    let doubles = [3.0, 0.5, 1.2, 0.6, 0.4, 1.0, 0.0, 1.4, 0.0, 0.3, 0.0, 0.2];
    let shells = device(&budget, &stream, &doubles);
    let short_shells = device(&budget, &stream, &doubles[1..]);
    // prim_begin, ao_offset, rank, rec_begin, bra, ket.
    let good = [0, 2, 3, 0, 1, 0, 0, 0, 2, 0, 1];
    let tables = || pinned(&budget, &good);
    let mut stray = good;
    stray[9] = 2; // the bra past the two shells
    let mut bad_prefix = good;
    bad_prefix[8] = 3; // three records for a pair of 2 × 1 primitives
    let len = good.len();
    let mut tables_dev = filled::<i32>(&budget, &stream, len);
    let mut short_dev = filled::<i32>(&budget, &stream, len - 1);
    let mut f64s = filled::<f64>(&budget, &stream, RECORD_F64_FIELDS * n_rec);
    let mut short_f64s = filled::<f64>(&budget, &stream, RECORD_F64_FIELDS * n_rec - 1);
    let mut i32s = filled::<i32>(&budget, &stream, RECORD_I32_FIELDS * n_rec);
    let mut short_i32s = filled::<i32>(&budget, &stream, RECORD_I32_FIELDS * n_rec - 1);
    stream.sync().unwrap();

    let mut messages = Vec::new();
    let sizes = ((n_shell, n_prim), (n_pairs, n_rec));
    messages.push(refusal(|| {
        let _ = mmd_records(
            &stream,
            sizes.0,
            &short_shells,
            sizes.1,
            tables(),
            &mut tables_dev,
            &mut f64s,
            &mut i32s,
        );
    }));
    messages.push(refusal(|| {
        let short = pinned(&budget, &good[1..]);
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            short,
            &mut tables_dev,
            &mut f64s,
            &mut i32s,
        );
    }));
    messages.push(refusal(|| {
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            tables(),
            &mut short_dev,
            &mut f64s,
            &mut i32s,
        );
    }));
    messages.push(refusal(|| {
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            tables(),
            &mut tables_dev,
            &mut short_f64s,
            &mut i32s,
        );
    }));
    messages.push(refusal(|| {
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            tables(),
            &mut tables_dev,
            &mut f64s,
            &mut short_i32s,
        );
    }));
    messages.push(refusal(|| {
        let stray = pinned(&budget, &stray);
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            stray,
            &mut tables_dev,
            &mut f64s,
            &mut i32s,
        );
    }));
    messages.push(refusal(|| {
        let bad = pinned(&budget, &bad_prefix);
        let _ = mmd_records(
            &stream,
            sizes.0,
            &shells,
            sizes.1,
            bad,
            &mut tables_dev,
            &mut f64s,
            &mut i32s,
        );
    }));
    assert!(
        messages[5].contains("pair 0 names shells 2 and 1 of 2"),
        "the stray pair is named: {}",
        messages[5]
    );
    assert!(
        messages[6].contains("pair 0's records [0, 3) are not its primitive product"),
        "the bad prefix is named: {}",
        messages[6]
    );
    nothing_enqueued(
        "fb_mmd_records",
        &messages,
        &[
            ("f64_fields", read(&budget, &stream, &f64s)),
            ("the short f64_fields", read(&budget, &stream, &short_f64s)),
        ],
        &[
            ("tables_dev", read(&budget, &stream, &tables_dev)),
            ("the short tables_dev", read(&budget, &stream, &short_dev)),
            ("i32_fields", read(&budget, &stream, &i32s)),
            ("the short i32_fields", read(&budget, &stream, &short_i32s)),
        ],
    );
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_gather_refuses_a_short_slice_or_a_stray_index_before_anything_is_enqueued() {
    let ledger = ledger();
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let (n_ao, n_columns, b) = (3, 2, 1);
    // Column 0 at (0, 1), off-diagonal; column 1 at (2, 2), diagonal.
    let good = [0, 2, 1, 2];
    let tables = || pinned(&budget, &good);
    let mut stray = good;
    stray[1] = 3; // column 1's row past the matrix
    let weights = device(&budget, &stream, &[2.0, 1.0]);
    let short_weights = device(&budget, &stream, &[2.0]);
    let packed = device(&budget, &stream, &[0.5, -1.5]);
    let short_packed = device(&budget, &stream, &[0.5]);
    let mut tables_dev = filled::<i32>(&budget, &stream, good.len());
    let mut short_dev = filled::<i32>(&budget, &stream, good.len() - 1);
    let mut slab = filled::<f64>(&budget, &stream, n_ao * n_ao);
    let mut short_slab = filled::<f64>(&budget, &stream, n_ao * n_ao - 1);
    stream.sync().unwrap();

    let sizes = (n_ao, n_columns, b);
    let mut messages = Vec::new();
    messages.push(refusal(|| {
        let short = pinned(&budget, &good[1..]);
        let _ = packed_gather(
            &stream,
            sizes,
            short,
            &mut tables_dev,
            &weights,
            &packed,
            &mut slab,
        );
    }));
    messages.push(refusal(|| {
        let _ = packed_gather(
            &stream,
            sizes,
            tables(),
            &mut short_dev,
            &weights,
            &packed,
            &mut slab,
        );
    }));
    messages.push(refusal(|| {
        let _ = packed_gather(
            &stream,
            sizes,
            tables(),
            &mut tables_dev,
            &short_weights,
            &packed,
            &mut slab,
        );
    }));
    messages.push(refusal(|| {
        let _ = packed_gather(
            &stream,
            sizes,
            tables(),
            &mut tables_dev,
            &weights,
            &short_packed,
            &mut slab,
        );
    }));
    messages.push(refusal(|| {
        let _ = packed_gather(
            &stream,
            sizes,
            tables(),
            &mut tables_dev,
            &weights,
            &packed,
            &mut short_slab,
        );
    }));
    messages.push(refusal(|| {
        let stray = pinned(&budget, &stray);
        let _ = packed_gather(
            &stream,
            sizes,
            stray,
            &mut tables_dev,
            &weights,
            &packed,
            &mut slab,
        );
    }));
    assert!(
        messages[5].contains("column 1 stands at (3, 2), outside a 3 × 3 matrix"),
        "the stray column is named: {}",
        messages[5]
    );
    nothing_enqueued(
        "fb_packed_gather",
        &messages,
        &[
            ("slab", read(&budget, &stream, &slab)),
            ("the short slab", read(&budget, &stream, &short_slab)),
        ],
        &[
            ("tables_dev", read(&budget, &stream, &tables_dev)),
            ("the short tables_dev", read(&budget, &stream, &short_dev)),
        ],
    );

    // The same call on good inputs runs: (0, 1) and (1, 0) hold column 0,
    // (2, 2) column 1, and the rest is zero.
    let tables = packed_gather(
        &stream,
        sizes,
        tables(),
        &mut tables_dev,
        &weights,
        &packed,
        &mut slab,
    )
    .unwrap();
    drop(tables.sync().unwrap());
    let got = read(&budget, &stream, &slab);
    assert_eq!(got, [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.5]);
}
