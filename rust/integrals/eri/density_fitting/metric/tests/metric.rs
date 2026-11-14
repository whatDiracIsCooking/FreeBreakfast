//! Each fixture run once on a card (tier 4, device 0) against PySCF's
//! `int2c2e` over the same fitting-set document, written by
//! `tests/data/reference.py` with the tree's basis digits, its CODATA 2018
//! bohr and its unrenormalized contractions (`D M D`), so the comparison
//! measures the integrals and nothing else.
//!
//! The bar is the C++ tree's for the same compare
//! (`test/gtest/df_metric_gpu.cpp`, FelizJueves's own against libcint,
//! metric.md "Tolerances"): `max |M - M_ref| <= 1e-12 max |M_ref|`, one
//! run. Its analysis: each element is a sum over the products of a few
//! primitives, the Hermite recursion and the Boys tower each near an ulp,
//! then two GEMMs over at most 28 Cartesian functions per shell block; the
//! C++ tree measured 1.4e-14 on these sets and 1.2e-14 through (i|i)
//! (`df_high_aux_gpu.cpp`), two decades inside it.
//!
//! And the average's promise, exactly: `M[P][Q]` and `M[Q][P]` carry the
//! same bits.

mod common;

/// The bar, relative to `max |M_ref|`.
const BAR: f64 = 1e-12;

/// Each fixture planned (its largest momentum the one it is chosen to
/// reach: f, and i), run once on device 0, compared with PySCF's M
/// against the bar, and checked exactly symmetric; its largest error, its
/// largest momentum and its task times are printed, and every miss named
/// before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn metric_matches_pyscf_and_is_exactly_symmetric() {
    let mut misses = Vec::new();
    for (stem, xyz, aux, l_max) in common::FIXTURES {
        let fixture = common::plan(xyz, aux);
        assert_eq!(fixture.l_max, l_max, "{stem}: the classes the case reaches");
        let n = fixture.n_aux;
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
        let got = std::mem::take(&mut *fixture.m.lock().unwrap());

        let want = fb_io_npy::read_npy(&common::data(&format!("{stem}.npy"))).unwrap();
        assert_eq!(want.shape, [n, n], "{stem}: the fitting-function count");
        assert_eq!(got.len(), n * n, "{stem}: M downloaded");
        let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want.data)
            .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{stem}: n_aux {n}, l <= {}, max |M| {scale:.6}, max |M - M_ref| / max |M| = {:e}, \
             bar {BAR:e}",
            fixture.l_max,
            worst / scale
        );
        if worst > BAR * scale {
            misses.push(format!(
                "{stem}: max |M - M_ref| = {worst:e} > {:e}",
                BAR * scale
            ));
        }
        let asymmetric = (0..n)
            .flat_map(|q| (q + 1..n).map(move |p| (p, q)))
            .filter(|&(p, q)| got[p + n * q].to_bits() != got[q + n * p].to_bits())
            .count();
        if asymmetric > 0 {
            misses.push(format!(
                "{stem}: {asymmetric} pairs M[P][Q], M[Q][P] differ in their bits"
            ));
        }
    }
    assert!(misses.is_empty(), "failed:\n{}", misses.join("\n"));
}
