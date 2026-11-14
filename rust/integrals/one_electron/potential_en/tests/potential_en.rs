//! Each fixture run once on a card (tier 4, device 0) against PySCF's V
//! (`int1e_nuc`, point nuclei), written by `tests/data/reference.py` with
//! the tree's basis digits, its CODATA 2018 bohr (the nuclei's too) and its
//! unrenormalized contractions (`D V D`), so the comparison measures the
//! integrals and nothing else. The C++ `core_hamiltonian_gpu.cpp` holds its
//! V to libcint's at the same bar.
//!
//! The bar is fixed by analysis, one run:
//! `max |V - V_ref| <= 1e-12 * max |V_ref|`, the rounding bound of the
//! atomic sums over primitive pairs and nuclei, the kernels' Boys evaluation
//! and the two GEMMs at these sizes (at most 60 spherical and 70 Cartesian
//! AOs, two nuclei) with two decades of margin. Symmetry is not gated: the
//! reference is symmetric.

mod common;

/// The bar, relative to `max |V_ref|`.
const BAR: f64 = 1e-12;

/// Each fixture planned, run once on device 0 and compared with PySCF's V
/// against the bar; its largest error and its task times are printed, and
/// every miss named before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn nuclear_attraction_matches_pyscf() {
    let mut misses = Vec::new();
    for (stem, xyz, basis) in common::FIXTURES {
        let fixture = common::plan(xyz, basis);
        let n = fixture.n_ao;
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
        let got = std::mem::take(&mut *fixture.v.lock().unwrap());

        let want = fb_io_npy::read_npy(&common::data(&format!("{stem}.npy"))).unwrap();
        assert_eq!(want.shape, [n, n], "{stem}: the AO count");
        assert_eq!(got.len(), n * n, "{stem}: V downloaded");
        let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want.data)
            .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{stem}: n_ao {n}, max |V - V_ref| = {worst:e}, bar {:e}",
            BAR * scale
        );
        if worst > BAR * scale {
            misses.push(format!(
                "{stem}: max |V - V_ref| = {worst:e} > {:e}",
                BAR * scale
            ));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}
