//! The factor against the density it factors: `2 L Lᵀ` is
//! `fb_sad::sad_density` to rounding, each stamp's rank is its atom's count
//! of occupied orbitals with a clear gap below it, and L's width is the
//! atoms' ranks summed.

use std::path::PathBuf;

use fb_sad_factor::{RANK_TOLERANCE, density_of, sad_factor, stamp_factor};

/// The bar on `max |2 L Lᵀ - D|`, absolute; D's elements are of order 1.
const BAR: f64 = 1e-12;

/// A molecule of the tree's fixtures.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../rust/steps/tests/data")
        .join(file)
}

/// The occupied orbitals of a neutral atom's spherical average: one per m
/// of each subshell the Madelung fill reaches.
fn atomic_rank(z: i32) -> usize {
    match z {
        1 | 2 => 1,
        3 | 4 => 2,
        5..=10 => 5,
        _ => panic!("no fixture has Z={z}"),
    }
}

fn check(xyz: &str, basis_name: &str) {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis(basis_name)
        .expect("a carried basis")
        .primitive_normalized()
        .place(&molecule)
        .expect("the basis covers the molecule");
    let baked = Some(fb_io_sad::builtin_sad_density as fb_sad::SadStampLookup);
    let factor = sad_factor(&basis, molecule.atoms(), baked).expect("the factor");
    let n = factor.n_ao;

    let mut rank = 0;
    for element in &factor.elements {
        let f = &element.factor;
        assert_eq!(f.rank, atomic_rank(element.z), "{xyz}: Z={}", element.z);
        assert!(
            f.largest_dropped < 1e-3 * RANK_TOLERANCE * f.smallest_kept,
            "{xyz}: Z={} has no clear gap: kept {:e}, dropped {:e}",
            element.z,
            f.smallest_kept,
            f.largest_dropped
        );
        rank += f.rank * element.atoms;
    }
    assert_eq!(factor.rank, rank, "{xyz}: L's width");
    assert_eq!(factor.columns.len(), n * rank);

    let want = fb_sad::sad_density(&basis, molecule.atoms(), baked).expect("the density");
    let got = density_of(&factor.columns, n, rank);
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "{xyz} {basis_name}: {n} AOs, rank {rank}, {} occupied, max |2 L Lt - D| = {worst:.2e}",
        molecule.electron_count() / 2
    );
    assert!(worst <= BAR, "{xyz} {basis_name}: {worst:e} past {BAR:e}");
}

#[test]
fn gly5_sto3g() {
    check("gly5.xyz", "sto-3g");
}

#[test]
fn gly5_cc_pvdz() {
    check("gly5.xyz", "cc-pvdz");
}

#[test]
fn gly10_cc_pvdz() {
    check("gly10.xyz", "cc-pvdz");
}

/// A rank-2 matrix of side 4 with known eigenvalues: the factor keeps two
/// columns, the largest's first, and reproduces it.
#[test]
fn a_known_stamp() {
    let n = 4;
    let u = [0.5, 0.5, 0.5, 0.5];
    let w = [0.5, -0.5, 0.5, -0.5];
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + n * j] = 3.0 * u[i] * u[j] + 0.5 * w[i] * w[j];
        }
    }
    let f = stamp_factor(&d, n);
    assert_eq!(f.rank, 2);
    assert!((f.smallest_kept - 0.5).abs() < 1e-14);
    assert!(f.largest_dropped < 1e-15);
    // Column 0 is u sqrt(3/2), up to sign.
    for i in 0..n {
        assert!((f.columns[i].abs() - 0.5 * 1.5_f64.sqrt()).abs() < 1e-14);
    }
    let back = density_of(&f.columns, n, f.rank);
    for (g, w) in back.iter().zip(&d) {
        assert!((g - w).abs() < 1e-14);
    }
}
