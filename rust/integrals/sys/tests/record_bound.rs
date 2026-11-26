//! The v0.0.4 layer's record bound, `record_bound` (tier 4): on one s
//! shell pair of one primitive each, on two centres, the dressed bound is
//! `sqrt((ab|ab))` of the primitive product, in closed form; a records
//! slice one field short is refused before anything is enqueued.
//!
//! The product of two s primitives is `c_a c_b K exp(-ζ |r - P|²)`,
//! `K = exp(-μ R²)`, `μ = αβ/ζ`, whose Coulomb self-energy is
//! `(c_a c_b K)² 2 π^{5/2} / (ζ² sqrt(2ζ))`: so
//! `sqrt((ab|ab)) = |c_a c_b| K 2^{1/4} π^{5/4} ζ^{-5/4}`. The kernel forms
//! the same from `s = c_a c_b (π/ζ)^{3/2}`, `E^{00}_0 = K` and
//! `I_0 = 2^{1/4} π^{-1/4} ζ^{1/4}`, a handful of roundings: the bar is
//! 1e-14 relative.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use fb_core_budget::{Backing, Budget, Caps, DeviceBuf, Ledger, PinnedBuf, Plain};
use fb_core_device::{Stream, part};
use fb_integrals_sys::{RECORD_F64_FIELDS, RECORD_I32_FIELDS, mmd_records, record_bound};

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

#[test]
#[ignore = "tier 4: needs a GPU"]
fn an_s_pairs_bound_is_its_closed_form_and_a_short_slice_is_refused() {
    let ledger = ledger();
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let (alpha, beta, ca, cb) = (1.3, 0.4, 0.7, -0.9);
    let (a, b) = ([0.0, 0.0, 0.0], [0.3, -0.5, 1.1]);
    // Exponents, coefficients, then the centres' x, y and z.
    let shells = device(
        &budget,
        &stream,
        &[alpha, beta, ca, cb, a[0], b[0], a[1], b[1], a[2], b[2]],
    );
    // prim_begin, ao_offset, rank, rec_begin, bra, ket: the pair (0, 1).
    let tables = pinned(&budget, &[0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]);
    let mut tables_dev = budget.device::<i32>(0, 11);
    let mut f64s = budget.device::<f64>(0, RECORD_F64_FIELDS);
    let mut i32s = budget.device::<i32>(0, RECORD_I32_FIELDS);
    let records = mmd_records(
        &stream,
        (2, 2),
        &shells,
        (1, 1),
        tables,
        &mut tables_dev,
        &mut f64s,
        &mut i32s,
    )
    .unwrap();
    let mut bound = budget.device::<f64>(0, 1);
    record_bound(&stream, (0, 0), 1, &f64s, &mut bound).unwrap();
    drop(records.sync().unwrap());
    let host = budget.pinned::<f64>(0, 1);
    let got = stream.download(host, &bound).unwrap().sync().unwrap()[0];

    let zeta = alpha + beta;
    let r2: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    let pi = std::f64::consts::PI;
    let want = (ca * cb).abs()
        * (-alpha * beta / zeta * r2).exp()
        * 2f64.powf(0.25)
        * pi.powf(1.25)
        * zeta.powf(-1.25);
    println!("s pair bound {got:e}, closed form {want:e}");
    assert!(
        (got - want).abs() <= 1e-14 * want,
        "{got:e} against {want:e}"
    );

    let short = part(&f64s, 0, 10);
    let message = catch_unwind(AssertUnwindSafe(|| {
        let _ = record_bound(&stream, (0, 0), 1, &short, &mut bound);
    }))
    .expect_err("a records slice one field short is refused");
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| message.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    assert!(message.contains("fb_ordered_pair_bound"), "{message}");
}
