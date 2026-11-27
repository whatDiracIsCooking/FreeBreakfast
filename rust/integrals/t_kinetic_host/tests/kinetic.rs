//! The host kinetic energy, by both routes, against PySCF's `int1e_kin`, on
//! its fixtures and references
//! (`rust/integrals/t_kinetic_host/tests/data/`, whose `reference.py`
//! pins the basis digits, the CODATA 2018 bohr and the unrenormalized
//! contractions): water in STO-3G and cc-pVDZ, and CO in cc-pVTZ, f on both
//! atoms and the bond off every axis. The packed lower triangle is compared
//! with the reference's, at the bar, `1e-12 max |T_ref|`.

use std::path::{Path, PathBuf};

use fb_core_primitives::lower_index;
use fb_integrals_t_kinetic_host::{kinetic, kinetic_via_cartesian};

/// The bar, relative to `max |T_ref|`.
const BAR: f64 = 1e-12;

/// A file of `tests/data/`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
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

/// `got`, a packed lower triangle, against the lower triangle of the full
/// `n × n` `reference`, at [`BAR`]; the largest error printed.
fn compare(label: &str, got: &[f64], reference: &Path, n: usize) {
    let want = fb_io_npy::read_npy(reference).expect("the reference reads");
    assert_eq!(want.shape, [n, n], "{label}: the AO count");
    assert_eq!(got.len(), lower_index(n, 0), "{label}");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for p in 0..n {
        for q in 0..=p {
            worst = worst.max((got[lower_index(p, q)] - want.data[p + q * n]).abs());
        }
    }
    println!(
        "{label}: nao {n}, max |got - ref| / max |ref| = {:e}, bar {BAR:e}",
        worst / scale
    );
    assert!(
        worst <= BAR * scale,
        "{label}: max |got - ref| = {worst:e} > {:e}",
        BAR * scale
    );
}

/// One fixture: T's lower triangle computed by each route, each compared
/// with PySCF.
fn check(stem: &str, xyz: &str, basis_name: &str) {
    let basis = basis(xyz, basis_name);
    let n = basis.nao_sphe() as usize;
    let reference = data(&format!("{stem}.npy"));
    let direct = kinetic(&basis);
    compare(&format!("{stem}, T spherical"), &direct, &reference, n);
    let via = kinetic_via_cartesian(&basis);
    compare(&format!("{stem}, T via cartesian"), &via, &reference, n);
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
    let (t1, t8) = (on(1), on(8));
    assert!(t1.iter().zip(&t8).all(|(x, y)| x.to_bits() == y.to_bits()));
}
