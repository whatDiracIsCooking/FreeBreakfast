//! The fixture the test binaries plan: water in cc-pVDZ, its records
//! ([`fb_integrals_mmd::add_records`]) and the step over cc-pVDZ-JKFIT at
//! a screen, on worker 0 with its BLAS workspace declared, as the binary
//! plans it.

use std::path::PathBuf;

use fb_basis::BasisSet;
use fb_core_planner::Dag;
use fb_molecule::Molecule;
use fb_pipeline_eri::{DensityFitted, add_df_eri};

/// The fitting set the fixture names.
pub const AUX: &str = "cc-pvdz-jkfit";

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// Water, and cc-pVDZ placed on it as the binary places it (charge and
/// multiplicity enter no integral).
pub fn water() -> (Molecule, BasisSet) {
    molecule("h2o.xyz")
}

/// The molecule of `xyz` in `tests/data/`, and cc-pVDZ placed on it.
pub fn molecule(xyz: &str) -> (Molecule, BasisSet) {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers water");
    (molecule, basis)
}

/// The records and the step over `aux` at `threshold`, added to `dag`.
pub fn add_step(
    dag: &mut Dag<'_>,
    molecule: &Molecule,
    basis: &BasisSet,
    aux: &str,
    threshold: f64,
) -> fb_core_error_handling::Result<DensityFitted> {
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(dag, 0, basis).expect("the library loads");
    add_df_eri(dag, 0, molecule, basis, &classes, &mut None, aux, threshold)
}
