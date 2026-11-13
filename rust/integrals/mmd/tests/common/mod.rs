//! The two molecules both test binaries plan: H2O/cc-pVDZ and CO/cc-pVTZ,
//! their bases as the BasisSet step makes them (the compiled-in set, its
//! primitives normalized, placed on the atoms).
//!
//! The geometries, in bohr, are the C++ tree's mmd fixtures
//! (`test/gtest/data/coulomb/{h2o_ccpvdz,co_ccpvtz}.json`), copied, and
//! the same literals `tools/mmd_records/main.cpp` dumps the reference
//! records from: no atom on an axis.

use fb_atom::{Atom, Element};
use fb_basis::BasisSet;
use fb_core_types::Double3;
use fb_molecule::Molecule;

/// A molecule of `atoms` (bohr) in the compiled-in set `basis`.
fn placed(basis: &str, atoms: &[(Element, [f64; 3])]) -> BasisSet {
    let molecule = Molecule::new(
        atoms
            .iter()
            .map(|&(element, [x, y, z])| Atom {
                position: Double3 { x, y, z },
                element,
            })
            .collect(),
        0,
        1,
    );
    let stamp = fb_io_basis::resolve_basis(basis).expect("a compiled-in set");
    stamp
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers the molecule")
}

/// Each case: its file stem, beside the goldens and the reference data, and
/// its basis.
pub fn cases() -> [(&'static str, BasisSet); 2] {
    [
        (
            "h2o_ccpvdz",
            placed(
                "cc-pvdz",
                &[
                    (Element::O, [0.0, 0.0, 0.0]),
                    (Element::H, [1.43, 0.21, -1.07]),
                    (Element::H, [-1.38, 0.35, -1.12]),
                ],
            ),
        ),
        (
            "co_ccpvtz",
            placed(
                "cc-pvtz",
                &[
                    (Element::C, [0.13, -0.21, 0.17]),
                    (Element::O, [0.77, 0.86, 1.86]),
                ],
            ),
        ),
    ]
}
