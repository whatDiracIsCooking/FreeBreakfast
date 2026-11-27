//! `src/molecule` (module `free_breakfast.molecule`), its host half: a
//! geometry as host data, its electron count, and the nuclear repulsion
//! energy.
//!
//! Everything is in atomic units: positions in bohr, energies in hartree,
//! a nucleus named by its element (Z). In Hartree atomic units the Coulomb
//! prefactor is exactly 1, so the repulsion is a bare `Σ Z_A Z_B / R_AB`.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `distance` | [`distance`] |
//! | `nuclear_charge` | [`nuclear_charge`] |
//! | `Molecule(res, atoms, charge = 0, multiplicity = 1)` | [`Molecule::new`]`(atoms, charge, multiplicity)`, aborting on the same inputs |
//! | `atoms`, `natoms`, `charge`, `multiplicity`, `total_nuclear_charge`, `electron_count` | the [`Molecule`] methods of those names |
//! | `nuclear_repulsion_energy` | [`nuclear_repulsion_energy`] |
//! | the nuclear repulsion gradient and Hessian | nothing yet: ported with the first gradient |
//! | the nuclei on the device (`Molecule`'s `upload`, `view`, `device_stream`), `nuclei_view.h`'s `NucleiView` | nothing: the nuclei stay on the host, below |
//! | `molecule_footprint` | nothing: there is no device copy to count |
//! | the re-export of `free_breakfast.atom` and `free_breakfast.core.types` | nothing: a dependent names `fb_atom` and `fb_core_types` itself |
//!
//! **The nuclei stay on the host.** The C++ `Molecule` uploads Z and the
//! positions at construction, from buffers it allocates itself, so building
//! one needs a GPU and the type is movable, not copyable. Here a `Molecule`
//! is host data alone, `Clone`, built on a device-less box, and this crate
//! links nothing. No step needs the nuclei on the device: the core
//! Hamiltonian step reads them on the host
//! (`fb_integrals_v_potential_en_host::point_charges`).
//!
//! **No default arguments.** Rust has none, so [`Molecule::new`] takes the
//! charge and multiplicity every time; a neutral singlet is `(atoms, 0, 1)`.
//!
//! The three repulsion functions run the C++ loops in the same order, so
//! each result is the same double.

#![forbid(unsafe_code)]

use fb_atom::Atom;
use fb_core_error_handling::{Error, check_or_abort};
use fb_core_types::Double3;

/// The Euclidean distance between two points, in bohr: `distance`.
pub fn distance(a: Double3, b: Double3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    ((dx * dx) + (dy * dy) + (dz * dz)).sqrt()
}

/// The atomic number Z of `atom`, its nuclear charge: `nuclear_charge`.
pub fn nuclear_charge(atom: &Atom) -> i32 {
    i32::from(atom.element.z())
}

/// Atoms with a net charge and a spin multiplicity, fixed at construction:
/// `Molecule`, without its nuclei on the device (the crate doc says where
/// they go).
///
/// `charge` is in units of the elementary charge, positive for a cation;
/// `multiplicity` is 2S + 1.
#[derive(Clone, Debug)]
pub struct Molecule {
    atoms: Vec<Atom>,
    charge: i32,
    multiplicity: i32,
}

impl Molecule {
    /// Hold `atoms` (positions in bohr), with `charge` and `multiplicity`.
    ///
    /// No atoms, or more than an `i32` counts, aborts at the caller with a
    /// critical line, as the C++ constructor does. Input that can be empty,
    /// an XYZ file, is refused as an [`Error`] by its reader before a
    /// `Molecule` is built.
    #[track_caller]
    pub fn new(atoms: Vec<Atom>, charge: i32, multiplicity: i32) -> Molecule {
        if atoms.is_empty() || i32::try_from(atoms.len()).is_err() {
            check_or_abort::<()>(Err(Error::config(format!(
                "Molecule: a molecule of {} atoms",
                atoms.len()
            ))));
        }
        Molecule {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// The atoms, in the order supplied.
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// The number of nuclei.
    pub fn natoms(&self) -> u32 {
        // In range: `new` refuses more than an `i32` counts.
        self.atoms.len() as u32
    }

    /// The net charge, in units of the elementary charge.
    pub fn charge(&self) -> i32 {
        self.charge
    }

    /// The spin multiplicity, 2S + 1.
    pub fn multiplicity(&self) -> i32 {
        self.multiplicity
    }

    /// The total nuclear charge, `Σ Z_A`.
    pub fn total_nuclear_charge(&self) -> i32 {
        self.atoms.iter().map(nuclear_charge).sum()
    }

    /// The number of electrons, `Σ Z_A - charge`: removing an electron
    /// leaves a cation of charge +1, so the count falls as the charge rises.
    pub fn electron_count(&self) -> i32 {
        self.total_nuclear_charge() - self.charge
    }
}

/// The product of two atoms' charges, `Z_A Z_B`, as the C++ forms it.
fn charge_product(a: &Atom, b: &Atom) -> f64 {
    f64::from(nuclear_charge(a)) * f64::from(nuclear_charge(b))
}

/// The nuclear repulsion energy `E_nn = Σ_{A<B} Z_A Z_B / R_AB`, in
/// hartree: `nuclear_repulsion_energy`.
///
/// Each unordered pair is counted once; one atom has no pairs, and zero
/// energy.
pub fn nuclear_repulsion_energy(molecule: &Molecule) -> f64 {
    let atoms = molecule.atoms();
    let mut energy = 0.0;
    for (a, atom_a) in atoms.iter().enumerate() {
        for atom_b in &atoms[a + 1..] {
            let r = distance(atom_a.position, atom_b.position);
            energy += charge_product(atom_a, atom_b) / r;
        }
    }
    energy
}

/// `test/gtest/molecule_host.cpp`, and the cases of
/// `test/gtest/molecule_gpu.cpp` that need no device here, each named
/// after its C++ case. The abort case is `tests/molecule_abort.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use fb_atom::Element;

    fn at(x: f64, y: f64, z: f64, element: Element) -> Atom {
        Atom {
            position: Double3 { x, y, z },
            element,
        }
    }

    /// Water near its equilibrium geometry, in bohr.
    fn water(charge: i32, multiplicity: i32) -> Molecule {
        Molecule::new(
            vec![
                at(0.0, 0.0, 0.2217, Element::O),
                at(0.0, 1.4309, -0.8867, Element::H),
                at(0.0, -1.4309, -0.8867, Element::H),
            ],
            charge,
            multiplicity,
        )
    }

    /// `EXPECT_DOUBLE_EQ`: within four ulps.
    #[track_caller]
    fn assert_double_eq(actual: f64, expected: f64) {
        let ulps = (actual.to_bits() as i64 - expected.to_bits() as i64).unsigned_abs();
        assert!(ulps <= 4, "{actual} vs {expected}: {ulps} ulps");
    }

    #[test]
    fn distance_is_euclidean() {
        let origin = Double3::default();
        let p = Double3 {
            x: 3.0,
            y: 4.0,
            z: 12.0,
        };
        assert_double_eq(distance(origin, p), 13.0);
        let q = Double3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        assert_double_eq(distance(q, q), 0.0);
    }

    #[test]
    fn nuclear_charge_reads_the_atom() {
        assert_eq!(nuclear_charge(&at(1.5, -2.0, 0.25, Element::Cl)), 17);
    }

    #[test]
    fn electron_count_subtracts_charge() {
        assert_eq!(water(0, 1).total_nuclear_charge(), 10);
        assert_eq!(water(0, 1).electron_count(), 10);
        assert_eq!(water(1, 2).electron_count(), 9); // cation: one fewer electron
        assert_eq!(water(-1, 2).electron_count(), 11);
        assert_eq!(water(1, 2).multiplicity(), 2);
    }

    #[test]
    fn atoms_keep_supplied_order() {
        let mol = water(0, 1);
        assert_eq!(mol.natoms(), 3);
        assert_eq!(mol.atoms()[0].element, Element::O);
        assert_eq!(mol.atoms()[1].position.y, 1.4309);
        assert_eq!(mol.atoms()[2].position.y, -1.4309);
    }

    #[test]
    fn single_atom_has_no_repulsion() {
        let he = Molecule::new(vec![at(1.0, 2.0, 3.0, Element::He)], 0, 1);
        assert_eq!(nuclear_repulsion_energy(&he), 0.0);
    }

    #[test]
    fn hydrogen_molecule_repulsion() {
        let h2 = Molecule::new(
            vec![at(0.0, 0.0, 0.0, Element::H), at(0.0, 0.0, 1.4, Element::H)],
            0,
            1,
        );
        assert_double_eq(nuclear_repulsion_energy(&h2), 1.0 / 1.4);
    }

    #[test]
    fn water_repulsion_sums_every_pair_once() {
        // O-H twice (Z = 8 * 1) plus H-H once (Z = 1 * 1), each over its
        // own distance.
        let r_oh = 1.4309_f64.hypot(0.2217 + 0.8867);
        let r_hh = 2.0 * 1.4309;
        let expected = (2.0 * 8.0 / r_oh) + (1.0 / r_hh);
        assert!((nuclear_repulsion_energy(&water(0, 1)) - expected).abs() <= 1e-14);
        // Charge and multiplicity are about electrons; the nuclei do not care.
        assert_eq!(
            nuclear_repulsion_energy(&water(1, 2)),
            nuclear_repulsion_energy(&water(0, 1))
        );
    }
}
