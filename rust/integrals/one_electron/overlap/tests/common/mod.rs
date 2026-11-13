//! The three fixtures both test binaries run: H2O in STO-3G and cc-pVDZ,
//! CO in cc-pVTZ (f on both atoms, the bond off every axis), each an XYZ
//! file and a compiled-in basis, with PySCF's S beside them
//! (`tests/data/README.md`).

use std::path::PathBuf;

use fb_integrals_one_electron_overlap::Run;

/// Each fixture: its stem, naming its golden and its reference, its XYZ
/// file and its basis.
pub const FIXTURES: [(&str, &str, &str); 3] = [
    ("h2o_sto-3g", "h2o.xyz", "sto-3g"),
    ("h2o_cc-pvdz", "h2o.xyz", "cc-pvdz"),
    ("co_cc-pvtz", "co.xyz", "cc-pvtz"),
];

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The fixture's run, planned.
pub fn run(xyz: &str, basis: &str) -> Run {
    Run::new(&data(xyz), basis).expect("the fixture plans")
}
