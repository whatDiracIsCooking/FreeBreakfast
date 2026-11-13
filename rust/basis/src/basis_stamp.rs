//! `:basis_stamp`: the geometry-free basis definition, and placing it onto
//! a molecule.

use std::collections::BTreeMap;

use fb_core_error_handling::{Error, Result};
use fb_molecule::{Molecule, nuclear_charge};

use crate::{BasisSet, ShellStamp};

/// A geometry-free basis definition: for each element, by atomic number Z,
/// the shells an atom of that element wears, wherever it sits:
/// `BasisStamp`.
///
/// What a basis-set file specifies. [`BasisStamp::place`] stamps it onto a
/// molecule's atoms; one stamp placed at two geometries gives two consistent
/// [`BasisSet`]s. Ordered by Z, as the C++ `std::map` is. `Default` is the
/// empty stamp.
#[derive(Clone, Debug, Default)]
pub struct BasisStamp {
    element_shells: BTreeMap<i32, Vec<ShellStamp>>,
    name: String,
}

impl BasisStamp {
    /// The stamp over `element_shells`, Z to shells; elements a molecule
    /// does not use may be present. `name` is the compiled-in set this table
    /// is, or empty ([`BasisStamp::name`]).
    pub fn new(
        element_shells: BTreeMap<i32, Vec<ShellStamp>>,
        name: impl Into<String>,
    ) -> BasisStamp {
        BasisStamp {
            element_shells,
            name: name.into(),
        }
    }

    /// The canonical name of the compiled-in set this stamp is (`cc-pVDZ`),
    /// or empty. A name is a promise about the shells, so only the reader's
    /// compiled-in branch sets it; [`BasisStamp::place`] copies it onto the
    /// basis it makes.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The Z-to-shells table.
    pub fn element_shells(&self) -> &BTreeMap<i32, Vec<ShellStamp>> {
        &self.element_shells
    }

    /// The shells for element `z`, borrowed from the table, or `None` if the
    /// stamp carries none.
    pub fn shells_for(&self, z: i32) -> Option<&[ShellStamp]> {
        self.element_shells.get(&z).map(Vec::as_slice)
    }

    /// A copy with every shell's per-primitive normalization folded in
    /// ([`ShellStamp::primitive_normalized`]), and the name kept.
    #[must_use]
    pub fn primitive_normalized(&self) -> BasisStamp {
        let element_shells = self
            .element_shells
            .iter()
            .map(|(&z, shells)| {
                let normalized = shells.iter().map(ShellStamp::primitive_normalized);
                (z, normalized.collect())
            })
            .collect();
        BasisStamp {
            element_shells,
            name: self.name.clone(),
        }
    }

    /// This stamp on `molecule`'s atoms: for each atom, in molecule order,
    /// its element's shells at its position: `place(stamp, molecule)`.
    ///
    /// The AO layout follows atom order, then the stamp's shell order, and
    /// the basis records that grouping and the stamp's name. An atom whose
    /// element the stamp lacks is an [`fb_core_error_handling::ErrorKind::InvalidConfig`]
    /// error naming its Z: a basis that cannot cover the molecule is not a
    /// silently smaller basis.
    pub fn place(&self, molecule: &Molecule) -> Result<BasisSet> {
        let mut shells = Vec::new();
        let mut atom_shell_offsets = Vec::with_capacity(molecule.atoms().len() + 1);
        atom_shell_offsets.push(0);
        for atom in molecule.atoms() {
            let z = nuclear_charge(atom);
            let Some(element) = self.shells_for(z) else {
                return Err(Error::config(format!(
                    "basis has no shells for element Z={z}"
                )));
            };
            shells.extend(element.iter().map(|stamp| stamp.place(atom.position)));
            atom_shell_offsets.push(shells.len() as u32);
        }
        Ok(BasisSet::with_atoms(
            shells,
            atom_shell_offsets,
            self.name.clone(),
        ))
    }
}

/// `test/basis/basis_stamp_host.cpp`, and `test/gtest/basis_gpu.cpp`, which
/// needs no device here, one test per case, each named after it.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Primitive;
    use fb_atom::{Atom, Element};
    use fb_core_error_handling::ErrorKind;
    use fb_core_types::Double3;

    fn stamp(l: i32, exponent: f64) -> ShellStamp {
        ShellStamp::new(
            l,
            vec![Primitive {
                exponent,
                coefficient: 1.0,
            }],
        )
    }

    /// `basis_stamp_host.cpp`'s `water_stamp`.
    fn water_stamp(name: &str) -> BasisStamp {
        let mut table = BTreeMap::new();
        table.insert(1, vec![stamp(0, 1.2)]);
        table.insert(8, vec![stamp(0, 9.0), stamp(1, 2.0)]);
        BasisStamp::new(table, name)
    }

    /// `basis_gpu.cpp`'s `water_stamp`: carbon too, which no molecule uses.
    fn water_stamp_with_carbon() -> BasisStamp {
        let mut with_carbon = water_stamp("");
        with_carbon.element_shells.insert(6, vec![stamp(0, 5.0)]);
        with_carbon
    }

    fn at(x: f64, y: f64, z: f64, element: Element) -> Atom {
        Atom {
            position: Double3 { x, y, z },
            element,
        }
    }

    /// `BasisStamp.EmptyStampCarriesNothing`.
    #[test]
    fn empty_stamp_carries_nothing() {
        let stamp = BasisStamp::default();
        assert!(stamp.element_shells().is_empty());
        assert!(stamp.name().is_empty());
        assert!(stamp.shells_for(1).is_none());
    }

    /// `BasisStamp.ShellsForLooksUpByAtomicNumber`.
    #[test]
    fn shells_for_looks_up_by_atomic_number() {
        let stamp = water_stamp("");
        assert_eq!(stamp.element_shells().len(), 2);
        assert!(stamp.shells_for(7).is_none());
        let oxygen = stamp.shells_for(8).unwrap();
        assert_eq!(oxygen.len(), 2);
        assert_eq!(oxygen[0].angular_momentum(), 0);
        assert_eq!(oxygen[1].angular_momentum(), 1);
        assert_eq!(oxygen[1].primitives()[0].exponent, 2.0);
        // Borrowed from the stamp's own table, not a copy.
        assert!(std::ptr::eq(oxygen, stamp.element_shells()[&8].as_slice()));
    }

    /// `BasisStamp.NameIsKeptByPrimitiveNormalized`.
    #[test]
    fn name_is_kept_by_primitive_normalized() {
        assert!(water_stamp("").name().is_empty());
        let named = water_stamp("sto-3g");
        assert_eq!(named.name(), "sto-3g");
        assert_eq!(named.primitive_normalized().name(), "sto-3g");
    }

    /// `BasisStamp.PrimitiveNormalizedNormalizesEveryShell`: the stamp's
    /// normalization is the shell's, element by element and shell by shell,
    /// with the table's shape unchanged.
    #[test]
    fn primitive_normalized_normalizes_every_shell() {
        let raw = water_stamp("");
        let normalized = raw.primitive_normalized();
        assert_eq!(
            normalized.element_shells().len(),
            raw.element_shells().len()
        );
        for (&z, shells) in raw.element_shells() {
            let out = normalized.shells_for(z).expect("every Z kept");
            assert_eq!(out.len(), shells.len(), "Z = {z}");
            for (got, shell) in out.iter().zip(shells) {
                assert_eq!(*got, shell.primitive_normalized(), "Z = {z}");
            }
        }
        // The source is untouched: a unit coefficient stays one.
        assert_eq!(
            raw.shells_for(1).unwrap()[0].primitives()[0].coefficient,
            1.0
        );
        assert_ne!(
            normalized.shells_for(1).unwrap()[0].primitives()[0].coefficient,
            1.0
        );
    }

    /// `Basis.PlaceFollowsAtomOrderThenStampOrder`.
    #[test]
    fn place_follows_atom_order_then_stamp_order() {
        let water = Molecule::new(
            vec![
                at(0.0, 0.0, 0.2, Element::O),
                at(0.0, 1.4, -0.9, Element::H),
                at(0.0, -1.4, -0.9, Element::H),
            ],
            0,
            1,
        );
        let stamp = water_stamp_with_carbon();
        assert!(stamp.shells_for(8).is_some());
        assert!(stamp.shells_for(7).is_none());

        let basis = stamp.place(&water).unwrap();
        assert_eq!(basis.nshells(), 4);
        assert_eq!(basis.nao_cart(), 6); // O: s + p, then two H s
        let momenta = [0, 1, 0, 0];
        let center_y = [0.0, 0.0, 1.4, -1.4];
        for (s, shell) in basis.shells().iter().enumerate() {
            assert_eq!(shell.angular_momentum(), momenta[s]);
            assert_eq!(shell.center().y, center_y[s]);
        }
        assert_eq!(basis.natoms(), 3);
        let first_shell = [0, 2, 3];
        let shell_count = [2, 1, 1];
        for a in 0..3 {
            assert_eq!(basis.atom_shell_offset(a), first_shell[a as usize]);
            assert_eq!(basis.atom_shell_count(a), shell_count[a as usize]);
        }
    }

    /// `Basis.PlaceOntoUncoveredElementIsInvalidConfig`.
    #[test]
    fn place_onto_uncovered_element_is_invalid_config() {
        let nh = Molecule::new(
            vec![at(0.0, 0.0, 0.0, Element::N), at(0.0, 0.0, 1.9, Element::H)],
            0,
            1,
        );
        let error = water_stamp_with_carbon().place(&nh).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig);
        assert!(error.message.contains("Z=7"), "{}", error.message);
    }

    /// `Basis.PlacingOneStampAtTwoGeometriesAgreesOnLayout`, and the name
    /// carried onto the basis, which the C++ checks in neither binary.
    #[test]
    fn placing_one_stamp_at_two_geometries_agrees_on_layout() {
        let mut stamp = water_stamp_with_carbon().primitive_normalized();
        stamp.name = "sto-3g".to_owned();
        let a = Molecule::new(
            vec![at(0.0, 0.0, 0.0, Element::O), at(0.0, 0.0, 1.8, Element::H)],
            0,
            1,
        );
        let b = Molecule::new(
            vec![at(0.5, 0.0, 0.0, Element::O), at(0.5, 0.0, 2.1, Element::H)],
            0,
            1,
        );
        let ba = stamp.place(&a).unwrap();
        let bb = stamp.place(&b).unwrap();
        assert_eq!(ba.nao_cart(), bb.nao_cart());
        for s in 0..ba.nshells() {
            assert_eq!(ba.ao_offset_cart(s), bb.ao_offset_cart(s));
            let i = s as usize;
            assert_eq!(
                ba.shells()[i].primitives()[0].coefficient,
                bb.shells()[i].primitives()[0].coefficient
            );
        }
        assert_eq!(bb.shells()[0].center().x, 0.5);
        assert_eq!(ba.name(), "sto-3g");
    }
}
