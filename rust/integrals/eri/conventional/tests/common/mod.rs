//! What both test binaries plan: a molecule of `tests/data/` in a carried
//! basis, its records ([`fb_integrals_mmd::add_records`]) and the
//! conventional integrals over them, on worker 0 with its BLAS workspace
//! declared, as the binary plans them.

use std::path::{Path, PathBuf};

use fb_basis::BasisSet;
use fb_core_planner::Dag;
use fb_integrals_eri_conventional::{Conventional, add_conventional};

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The XYZ file `xyz` read, and the carried set `basis` placed on it
/// (charge and multiplicity enter no integral).
pub fn basis_on(xyz: &Path, basis: &str) -> BasisSet {
    let molecule = fb_io_xyz::read_xyz(xyz, 0, 1).expect("the XYZ file reads");
    fb_io_basis::resolve_basis(basis)
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers the molecule")
}

/// The records and the conventional integrals of `basis`, added to `dag`.
pub fn add_step(
    dag: &mut Dag<'_>,
    basis: &BasisSet,
) -> fb_core_error_handling::Result<Conventional> {
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(dag, 0, basis).expect("the library loads");
    add_conventional(dag, 0, basis, &classes, &mut None)
}
