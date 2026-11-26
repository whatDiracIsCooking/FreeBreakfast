//! The host overlap S against this crate's: the same matrix built on a card by
//! `fb_integrals_one_electron_overlap` and downloaded, on water and benzene in
//! cc-pVDZ (tier 4, device 0). The water's XYZ file is this crate's fixture,
//! benzene's `fb-pipeline-scf`'s.
//!
//! The bar is fixed by analysis, one run: `max |S - S_gpu| <= 1e-12 max
//! |S_gpu|`, every element of the full matrix against the host's packed lower
//! triangle. Each element is a sum of tens of FP64 terms (the primitive pairs
//! of a contracted pair, then a harmonic sum), which the two paths order
//! differently, so they differ by a few units in the last place of the largest
//! element, well inside the bar.

use std::path::PathBuf;

use fb_integrals_one_electron_overlap::Run;
use fb_integrals_s_overlap_host::{lower_index, overlap};

/// The bar, relative to `max |S_gpu|`.
const BAR: f64 = 1e-12;

/// Each fixture: a name, its XYZ file relative to `rust/`, and the set.
const FIXTURES: [(&str, &str, &str); 2] = [
    (
        "water",
        "integrals/one_electron/overlap/tests/data/h2o.xyz",
        "cc-pvdz",
    ),
    ("benzene", "pipeline/scf/tests/data/benzene.xyz", "cc-pvdz"),
];

/// A path relative to `rust/`, the nearest ancestor holding `Cargo.lock`.
fn rust(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|d| d.join("Cargo.lock").exists())
        .expect("a crate under rust/")
        .join(path)
}

/// `got`, a packed lower triangle, against `want`, the full `n × n` matrix
/// column-major: the largest error relative to `max |want|`, which must be
/// within [`BAR`].
fn compare(name: &str, got: &[f64], want: &[f64], n: usize) {
    assert_eq!(got.len(), lower_index(n, 0), "{name}: the host triangle");
    assert_eq!(want.len(), n * n, "{name}: the device matrix");
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for q in 0..n {
        for p in 0..n {
            let host = got[lower_index(p.max(q), p.min(q))];
            worst = worst.max((host - want[p + q * n]).abs());
        }
    }
    println!(
        "{name}: n {n}, max |S_gpu| {scale:.6}, max |S - S_gpu| / max |S_gpu| = {:e}, \
         bar {BAR:e}",
        worst / scale
    );
    assert!(
        worst <= BAR * scale,
        "{name}: max |S - S_gpu| = {worst:e} > {:e}",
        BAR * scale
    );
}

/// `set` placed on the molecule of `xyz`, primitive-normalized as the
/// pipeline places it.
fn setup(xyz: &str, set: &str) -> fb_basis::BasisSet {
    let molecule = fb_io_xyz::read_xyz(&rust(xyz), 0, 1).expect("the XYZ file reads");
    fb_io_basis::resolve_basis(set)
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers the molecule")
}

/// S on the host against S on device 0, each fixture. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn overlap_matches_the_device_build() {
    for (name, xyz, set) in FIXTURES {
        let basis = setup(xyz, set);
        let n = basis.nao_sphe() as usize;
        let (record, want) = Run::new(&rust(xyz), set)
            .expect("the device build plans")
            .run(0);
        record.result.unwrap();
        compare(name, &overlap(&basis), &want, n);
    }
}
