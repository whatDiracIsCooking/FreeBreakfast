//! The host nuclear attraction, by both routes, against PySCF's
//! `int1e_nuc`, on the device crate's own fixtures and references
//! (`rust/integrals/one_electron/potential_en/tests/data/`, whose README
//! says what they pin): water in STO-3G and cc-pVDZ, and CO in cc-pVTZ, f
//! on both atoms and the bond off every axis. The packed lower triangle is
//! compared with the reference's, at that crate's bar, `1e-12 max |V_ref|`.

use std::path::PathBuf;

use fb_integrals_v_potential_en_host::{
    PointCharge, lower_index, point_charges, potential_en, potential_en_via_cartesian,
};

/// The bar, relative to `max |V_ref|`.
const BAR: f64 = 1e-12;

/// A file of the device crate's `tests/data/`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../one_electron/potential_en/tests/data")
        .join(file)
}

/// `basis_name` placed on `xyz`, primitive-normalized as the pipeline does,
/// and the molecule's nuclei.
fn setup(xyz: &str, basis_name: &str) -> (fb_basis::BasisSet, Vec<PointCharge>) {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis(basis_name)
        .expect("a carried basis")
        .primitive_normalized()
        .place(&molecule)
        .expect("the basis covers the molecule");
    (basis, point_charges(&molecule))
}

/// One fixture: V's lower triangle computed by each route and compared;
/// its size and largest error printed.
fn check(stem: &str, xyz: &str, basis_name: &str) {
    let (basis, charges) = setup(xyz, basis_name);
    let n = basis.nao_sphe() as usize;
    let want = fb_io_npy::read_npy(&data(&format!("{stem}.npy"))).expect("the reference reads");
    assert_eq!(want.shape, [n, n], "{stem}: the AO count");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (route, got) in [
        ("one table", potential_en(&basis, &charges)),
        (
            "via cartesian",
            potential_en_via_cartesian(&basis, &charges),
        ),
    ] {
        assert_eq!(got.len(), lower_index(n, 0), "{stem}, {route}");
        let mut worst = 0.0_f64;
        for p in 0..n {
            for q in 0..=p {
                worst = worst.max((got[lower_index(p, q)] - want.data[p + q * n]).abs());
            }
        }
        println!(
            "{stem}, {route}: nao {n}, max |V_ref| {scale:.6}, \
             max |V - V_ref| / max |V_ref| = {:e}, bar {BAR:e}",
            worst / scale
        );
        assert!(
            worst <= BAR * scale,
            "{stem}, {route}: max |V - V_ref| = {worst:e} > {:e}",
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
    let (basis, charges) = setup("co.xyz", "cc-pvtz");
    let on = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool")
            .install(|| potential_en(&basis, &charges))
    };
    let (one, many) = (on(1), on(8));
    assert!(
        one.iter()
            .zip(&many)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
