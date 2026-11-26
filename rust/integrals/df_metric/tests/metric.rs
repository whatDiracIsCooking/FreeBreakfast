//! The host metric, by both routes, against PySCF's `int2c2e`, on its
//! fixtures and references
//! (`rust/integrals/df_metric/tests/data/`, whose README
//! says what they pin): water in cc-pVDZ-JKFIT, aux through f, and iron
//! beside water in def2-universal-JKFIT, aux through i with contracted
//! shells. The packed lower triangle is compared with the reference's, at
//! the bar, `1e-12 max |M_ref|`.

use std::path::PathBuf;

use fb_core_primitives::lower_index;
use fb_integrals_df_metric::{metric, metric_via_cartesian};

/// The bar, relative to `max |M_ref|`.
const BAR: f64 = 1e-12;

/// A file of `tests/data/`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// One fixture: M's lower triangle computed by each route and compared;
/// its size and largest error printed.
fn check(stem: &str, xyz: &str, aux_name: &str, l_max: i32) {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let aux = fb_io_basis::resolve_aux_basis(aux_name)
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers the molecule");
    let reached = aux
        .shells()
        .iter()
        .map(fb_basis::Shell::angular_momentum)
        .max();
    assert_eq!(
        reached,
        Some(l_max),
        "{stem}: the momentum the case reaches"
    );
    let n = aux.nao_sphe() as usize;

    let want = fb_io_npy::read_npy(&data(&format!("{stem}.npy"))).expect("the reference reads");
    assert_eq!(want.shape, [n, n], "{stem}: the fitting-function count");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (route, got) in [
        ("spherical", metric(&aux)),
        ("via cartesian", metric_via_cartesian(&aux)),
    ] {
        assert_eq!(
            got.len(),
            lower_index(n, 0),
            "{stem}, {route}: the packed length"
        );
        let mut worst = 0.0_f64;
        for p in 0..n {
            for q in 0..=p {
                let error = (got[lower_index(p, q)] - want.data[p + q * n]).abs();
                worst = worst.max(error);
            }
        }
        println!(
            "{stem}, {route}: n_aux {n}, l <= {l_max}, max |M| {scale:.6}, \
             max |M - M_ref| / max |M| = {:e}, bar {BAR:e}",
            worst / scale
        );
        assert!(
            worst <= BAR * scale,
            "{stem}, {route}: max |M - M_ref| = {worst:e} > {:e}",
            BAR * scale
        );
    }
}

#[test]
fn water_in_cc_pvdz_jkfit_matches_pyscf() {
    check("h2o_cc-pvdz-jkfit", "h2o.xyz", "cc-pvdz-jkfit", 3);
}

#[test]
fn iron_and_water_in_def2_universal_jkfit_matches_pyscf() {
    check(
        "feh2o_def2-universal-jkfit",
        "feh2o.xyz",
        "def2-universal-jkfit",
        6,
    );
}

/// The triangle carries the same bits on one thread as on rayon's default:
/// a shell pair's block depends on nothing another task computes.
#[test]
fn the_thread_count_does_not_change_a_bit() {
    let molecule = fb_io_xyz::read_xyz(&data("feh2o.xyz"), 0, 1).expect("the XYZ file reads");
    let aux = fb_io_basis::resolve_aux_basis("def2-universal-jkfit")
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers the molecule");
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool of one thread")
        .install(|| metric(&aux));
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("a pool of eight threads")
        .install(|| metric(&aux));
    assert!(
        one.iter()
            .zip(&many)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
