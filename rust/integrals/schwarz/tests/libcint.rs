//! The diagonal against libcint's, PySCF's `int2e_cart`, over water in
//! cc-pVTZ (`tests/data/README.md`): three centres, contracted shells
//! (general contractions among them) and every class through (ff|ff),
//! where the Boys function reaches order 12. No device.
//!
//! The bar is `1e-10` relative, element by element: the recursions'
//! accumulated rounding at L = 12 (the Hermite and R recursions a few
//! hundred operations deep, each near an ulp, and the quartet sum's
//! cancellation of mixed-sign Hermite terms) is well above fp64's 1e-16
//! per operation and well below any threshold the screen uses. One run.

use std::path::PathBuf;

/// The bar on `|got - want| / want`.
const BAR: f64 = 1e-10;

/// A file of `tests/data/`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// Every element of every shell pair `a <= b` of water in cc-pVTZ, in both
/// orders of the pair, within the bar of libcint's. Catches a Hermite
/// coefficient, an R recursion or a Boys order wrong in any class through
/// (ff|ff), a normalisation that is not the device kernels' (one shared
/// coefficient on every component), a block laid out with `n` slower, and
/// a quartet counted twice or not at all.
#[test]
fn water_cc_pvtz_matches_libcint_through_ff_ff() {
    let molecule = fb_io_xyz::read_xyz(&data("h2o.xyz"), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis("cc-pvtz")
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers water");
    let want = fb_io_npy::read_npy(&data("h2o_cc-pvtz.npy")).expect("the reference reads");
    let n = basis.nao_cart() as usize;
    assert_eq!(want.shape, [n, n], "the reference's function count");

    let ns = basis.nshells();
    let pairs: Vec<(u32, u32)> = (0..ns)
        .flat_map(|a| (a..ns).flat_map(move |b| [(a, b), (b, a)]))
        .collect();
    let got = fb_integrals_schwarz::schwarz_diagonal(&basis, &pairs);

    let mut at = 0;
    let (mut worst, mut misses) = (0.0_f64, Vec::new());
    for &(a, b) in &pairs {
        let (sa, sb) = (&basis.shells()[a as usize], &basis.shells()[b as usize]);
        let (oa, ob) = (
            basis.ao_offset_cart(a) as usize,
            basis.ao_offset_cart(b) as usize,
        );
        for m in 0..sa.function_count_cart() as usize {
            for k in 0..sb.function_count_cart() as usize {
                let w = want.data[(oa + m) + n * (ob + k)];
                let g = got[at];
                at += 1;
                let rel = (g - w).abs() / w;
                // Not f64::max, which would drop a NaN.
                if rel.is_nan() || rel > worst {
                    worst = rel;
                }
                if rel.is_nan() || rel > BAR {
                    misses.push(format!(
                        "shells ({a}, {b}), functions ({m}, {k}): {g:e} against {w:e}"
                    ));
                }
            }
        }
    }
    assert_eq!(at, got.len(), "every element read");
    println!(
        "water cc-pVTZ: {} pairs, {} elements, max relative difference {worst:e}, bar {BAR:e}",
        pairs.len(),
        got.len()
    );
    assert!(
        misses.is_empty(),
        "{} elements past the bar:\n{}",
        misses.len(),
        misses.join("\n")
    );
}
