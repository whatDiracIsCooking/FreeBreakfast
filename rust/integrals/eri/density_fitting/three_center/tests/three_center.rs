//! Each fixture run once on a card (tier 4, device 0) against PySCF's
//! `int3c2e` over the same documents, written by `tests/data/reference.py`
//! with the tree's basis digits, its CODATA 2018 bohr and its
//! unrenormalized contractions (`D_m D_n D_P`), so the comparison measures
//! the integrals and nothing else.
//!
//! The bar is the C++ tree's for the same compare
//! (`test/gtest/df_three_center_gpu.cpp` and `df_high_aux_gpu.cpp`,
//! FelizJueves's own against libcint, three_center.md "Tolerances"):
//! `max |Δ| <= 1e-12 max |(mn|P)_ref|`, one run. Its analysis: each element
//! is a sum over a few primitive triples, the Hermite recursion and the
//! Boys tower each near an ulp, then three GEMMs over at most 28 Cartesian
//! functions per shell block and the exact average; the C++ tree measured
//! 1.6e-14 of max through (ff|i), two decades inside it. The issue states
//! 1e-10; this is the tighter, cited one.
//!
//! And the average's promise, exactly: `(mn|P)` and `(nm|P)` carry the same
//! bits.

mod common;

use fb_basis::BasisSet;

/// The bar, relative to `max |(mn|P)_ref|`.
const BAR: f64 = 1e-12;

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// One fixture planned, run once on device 0, compared with PySCF's
/// tensor against the bar and checked exactly symmetric in (m, n); its
/// largest error and its task times are printed, and every miss named
/// before the test fails.
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
    assert_eq!(got.len(), n * n * n_aux, "{stem}: (mn|P) downloaded");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want.data)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "{stem}: nao {n}, naux {n_aux}, max |(mn|P)| {scale:.6}, max |Δ| / max |(mn|P)| = {:e}, \
         bar {BAR:e}",
        worst / scale
    );
    let mut misses = Vec::new();
    if worst > BAR * scale {
        misses.push(format!("max |Δ| = {worst:e} > {:e}", BAR * scale));
    }
    let at = |p: usize, a: usize, b: usize| got[p + n_aux * (b + n * a)];
    let asymmetric = (0..n)
        .flat_map(|m| (m + 1..n).map(move |k| (m, k)))
        .flat_map(|(m, k)| (0..n_aux).map(move |p| (p, m, k)))
        .filter(|&(p, m, k)| at(p, m, k).to_bits() != at(p, k, m).to_bits())
        .count();
    if asymmetric > 0 {
        misses.push(format!(
            "{asymmetric} elements (mn|P), (nm|P) differ in their bits"
        ));
    }
    assert!(misses.is_empty(), "{stem} failed:\n{}", misses.join("\n"));
}

/// Water in cc-pVDZ / cc-pVDZ-JKFIT, every class through (dd|f), against
/// PySCF, and exactly (m, n)-symmetric: the family's representative. Needs
/// a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_matches_pyscf_and_is_exactly_symmetric() {
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
