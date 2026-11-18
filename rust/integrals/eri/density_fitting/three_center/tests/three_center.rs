//! Each fixture's packed (mn|P), every pair kept as one batch, run once on
//! a card (tier 4, device 0) against PySCF's `int3c2e` over the same
//! documents, written by `tests/data/reference.py`
//! with the tree's basis digits, its CODATA 2018 bohr and its
//! unrenormalized contractions (`D_m D_n D_P`), so the comparison measures
//! the integrals and nothing else.
//!
//! The bar is the C++ tree's for the same compare
//! (`test/gtest/df_three_center_gpu.cpp` and `df_high_aux_gpu.cpp`,
//! FelizJueves's own against libcint, three_center.md "Tolerances"):
//! `max |Δ| <= 1e-12 max |(mn|P)_ref|`, one run. Its analysis: each element
//! is a sum over a few primitive triples, the Hermite recursion and the
//! Boys tower each near an ulp, then the per-class spherical GEMMs over at
//! most 28 Cartesian functions per shell block; the C++ tree measured
//! 1.6e-14 of max through (ff|i), two decades inside it. Every packed
//! column is compared, a diagonal block's upper columns included, each
//! with the reference's `(m, n)`. The square path's exact (m, n) average,
//! and its bitwise symmetry, went with it (#1721): each pair's `(m, n)` is
//! computed once here.

mod common;

use fb_basis::BasisSet;

/// The bar, relative to `max |(mn|P)_ref|`.
const BAR: f64 = 1e-12;

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// One fixture planned, run once on device 0, every packed column compared
/// with PySCF's tensor against the bar; its largest error and its task
/// times are printed.
fn check(stem: &str, basis: &BasisSet, aux: &BasisSet) {
    let fixture = common::plan(basis, aux);
    let (n, n_aux) = (fixture.n_ao, fixture.n_aux);
    let mut instance = fixture.planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    record.result.unwrap();
    for t in &record.tasks {
        println!(
            "{stem} {}@{}: host {:?}, device {:?}",
            t.name, t.worker, t.host, t.device
        );
    }
    let got = std::mem::take(&mut *fixture.out.lock().unwrap());

    let want = fb_io_npy::read_npy(&common::data(&format!("{stem}.npy"))).unwrap();
    assert_eq!(want.shape, [n_aux, n, n], "{stem}: the function counts");
    let blocks = fixture.map.sphe().blocks();
    assert_eq!(
        got.len(),
        n_aux * fixture.map.sphe().n_columns(),
        "{stem}: (mn|P) downloaded"
    );
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // Element (P, column) of the packed tensor at P + n_aux · column,
    // against the reference's (P, k + n m), every column of every block.
    let mut worst = 0.0_f64;
    for block in blocks {
        for a in 0..block.n[0] {
            for b in 0..block.n[1] {
                let column = block.column + a * block.n[1] + b;
                let (m, k) = (block.ao[0] + a, block.ao[1] + b);
                for p in 0..n_aux {
                    let d = (got[p + n_aux * column] - want.data[p + n_aux * (k + n * m)]).abs();
                    // Not f64::max, which would drop a NaN.
                    if d.is_nan() || d > worst {
                        worst = d;
                    }
                }
            }
        }
    }
    println!(
        "{stem}: nao {n}, naux {n_aux}, max |(mn|P)| {scale:.6}, max |Δ| / max |(mn|P)| = {:e}, \
         bar {BAR:e}",
        worst / scale
    );
    assert!(
        !worst.is_nan() && worst <= BAR * scale,
        "{stem}: max |Δ| = {worst:e} > {:e}",
        BAR * scale
    );
}

/// Water in cc-pVDZ / cc-pVDZ-JKFIT, every class through (dd|f), against
/// PySCF: the family's representative. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_matches_pyscf() {
    let (basis, aux) = common::water();
    check("h2o_cc-pvdz_cc-pvdz-jkfit", &basis, &aux);
}

/// The C++ tree's `df_high_aux` fixture, all 70 classes through (ff|i),
/// each with a one-centre and a two-centre pair, against PySCF at the same
/// bar: the orbital f and aux h and i rows no carried set reaches. A slow
/// case: it returns at once under `FREE_BREAKFAST_FAST_TESTS`
/// (`docs/testing.md`, "Slow cases"). Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn high_aux_matches_pyscf_through_ff_i() {
    if fast_tests() {
        eprintln!("skipped: slow case (FREE_BREAKFAST_FAST_TESTS is set)");
        return;
    }
    let (basis, aux) = common::high_aux();
    check("df_high_aux", &basis, &aux);
}
