//! The overlap's three fixtures, which both test binaries run: H2O in
//! STO-3G and cc-pVDZ, CO in cc-pVTZ (f on both atoms, the bond off every
//! axis), each an XYZ file and a compiled-in basis, read from
//! `integrals/one_electron/overlap/tests/data/` beside PySCF's S, with
//! PySCF's H in `integrals/one_electron/core_hamiltonian/tests/data/` and
//! the reference orbital energies in `tests/data/` here.

use std::path::PathBuf;

use fb_core_error_handling::Result;
use fb_pipeline_scf::Run;

/// Each fixture: its stem, naming its golden and its references, its XYZ
/// file and its basis.
pub const FIXTURES: [(&str, &str, &str); 3] = [
    ("h2o_sto-3g", "h2o.xyz", "sto-3g"),
    ("h2o_cc-pvdz", "h2o.xyz", "cc-pvdz"),
    ("co_cc-pvtz", "co.xyz", "cc-pvtz"),
];

/// A file of `rust/<dir>`, relative to the workspace root.
pub fn data(dir: &str, file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(dir)
        .join(file)
}

/// A fixture's XYZ file.
pub fn xyz(file: &str) -> PathBuf {
    data("integrals/one_electron/overlap/tests/data", file)
}

/// The fixture's run, planned with `lwork` for syevd's work.
pub fn run(xyz_file: &str, basis: &str, lwork: impl FnOnce(usize) -> Result<usize>) -> Run {
    Run::new(&xyz(xyz_file), basis, lwork).expect("the fixture plans")
}
