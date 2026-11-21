//! The host kinetic energy, by both routes, against PySCF's `int1e_kin`, on
//! the rebuild's GPU kinetic crate's own fixtures and references
//! (`rust/integrals/one_electron/kinetic/tests/data/`, whose `reference.py`
//! pins the basis digits, the CODATA 2018 bohr and the unrenormalized
//! contractions): water in STO-3G and cc-pVDZ, and CO in cc-pVTZ, f on both
//! atoms and the bond off every axis. The packed lower triangle is compared
//! with the reference's, at that crate's bar, `1e-12 max |T_ref|`.

use std::path::PathBuf;

use fb_t_kinetic_host::{kinetic, kinetic_via_cartesian, lower_index};

/// The bar, relative to `max |T_ref|`.
const BAR: f64 = 1e-12;

/// A file of the GPU crate's `tests/data/`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../rust/integrals/one_electron/kinetic/tests/data")
        .join(file)
}

/// `basis_name` placed on `xyz`, primitive-normalized as the pipeline does.
fn basis(xyz: &str, basis_name: &str) -> fb_basis::BasisSet {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    fb_io_basis::resolve_basis(basis_name)
        .expect("a carried basis")
        .primitive_normalized()
        .place(&molecule)
        .expect("the basis covers the molecule")
}

/// One fixture: T's lower triangle computed by each route and compared;
/// its size and largest error printed.
fn check(stem: &str, xyz: &str, basis_name: &str) {
    let basis = basis(xyz, basis_name);
    let n = basis.nao_sphe() as usize;
    let want = fb_io_npy::read_npy(&data(&format!("{stem}.npy"))).expect("the reference reads");
    assert_eq!(want.shape, [n, n], "{stem}: the AO count");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (route, got) in [
        ("spherical", kinetic(&basis)),
        ("via cartesian", kinetic_via_cartesian(&basis)),
    ] {
        assert_eq!(got.len(), lower_index(n, 0), "{stem}, {route}");
        let mut worst = 0.0_f64;
        for p in 0..n {
            for q in 0..=p {
                worst = worst.max((got[lower_index(p, q)] - want.data[p + q * n]).abs());
            }
        }
        println!(
            "{stem}, {route}: nao {n}, max |T - T_ref| / max |T_ref| = {:e}, bar {BAR:e}",
            worst / scale
        );
        assert!(
            worst <= BAR * scale,
            "{stem}, {route}: max |T - T_ref| = {worst:e} > {:e}",
            BAR * scale
        );
    }
}

#[test]
fn water_in_sto_3g_matches_pyscf() {
    check("h2o_sto-3g", "h2o.xyz", "sto-3g");
}

#[test]
fn water_in_cc_pvdz_matches_pyscf() {
    check("h2o_cc-pvdz", "h2o.xyz", "cc-pvdz");
}

#[test]
fn carbon_monoxide_in_cc_pvtz_matches_pyscf() {
    check("co_cc-pvtz", "co.xyz", "cc-pvtz");
}

/// The triangle carries the same bits on one thread as on eight: a shell
/// pair's block depends on nothing another task computes.
#[test]
fn the_thread_count_does_not_change_a_bit() {
    let basis = basis("co.xyz", "cc-pvtz");
    let on = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool")
            .install(|| kinetic(&basis))
    };
    let (one, many) = (on(1), on(8));
    assert!(
        one.iter()
            .zip(&many)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
