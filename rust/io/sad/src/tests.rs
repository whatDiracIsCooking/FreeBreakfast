//! `test/io/sad_host.cpp`'s fast cases, one test per case, each named after
//! it, and the configure check the C++ build makes by construction: every
//! `data/sad/` directory names a compiled-in basis set. Unit tests, so the
//! table stays private.
//!
//! The stamps are recomputed exactly as the generator computed them: the
//! compiled-in set through `resolve_basis`, its primitives normalized as
//! the BasisSet step normalizes them, the element's shells alone at the
//! origin, and `sad_density` with no lookup. The bar is bitwise, since
//! `fb-sad` runs the C++ generator's operations in the C++ order. The C++
//! slow case, every element through argon, is not here: the workspace has
//! no slow tier yet. One run each.

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, BasisStamp, Shell};
use fb_core_error_handling::ErrorKind;
use fb_core_types::Double3;
use fb_io_basis::resolve_basis;
use fb_sad::{SadStampLookup, sad_density};

use super::*;

/// The canonical names, as `BasisSet::name` holds them: `kCompiledInSets`.
const COMPILED_IN_SETS: [&str; 11] = [
    "STO-3G",
    "6-31G",
    "6-31G*",
    "6-31G**",
    "def2-SVP",
    "def2-TZVP",
    "cc-pVDZ",
    "cc-pVTZ",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "MINAO",
];

const ORIGIN: Double3 = Double3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

/// Element `z` alone at the origin, as one atom: the `Atom` the C++ test
/// builds.
fn lone_atom(z: i32) -> [Atom; 1] {
    [Atom {
        position: ORIGIN,
        element: Element::from_z(z as u8).unwrap(),
    }]
}

/// Element `z` of `normalized` alone at the origin, grouped as one atom:
/// `atom_basis`.
fn atom_basis(normalized: &BasisStamp, z: i32) -> BasisSet {
    let shells: Vec<Shell> = normalized
        .shells_for(z)
        .unwrap()
        .iter()
        .map(|shell| shell.place(ORIGIN))
        .collect();
    let count = shells.len() as u32;
    BasisSet::with_atoms(shells, vec![0, count], "")
}

/// Every baked stamp of `name` whose Z passes `keep`, against its SCF, bit
/// for bit: `expect_baked_matches_computed`.
fn expect_baked_matches_computed(name: &str, keep: impl Fn(i32) -> bool) {
    let normalized = resolve_basis(name)
        .unwrap_or_else(|e| panic!("{name}: {}", e.message))
        .primitive_normalized();
    for &z in normalized.element_shells().keys().filter(|&&z| keep(z)) {
        let computed = sad_density(&atom_basis(&normalized, z), &lone_atom(z), None)
            .unwrap_or_else(|e| panic!("{name} Z={z}: {}", e.message));
        let baked = builtin_sad_density(name, z).unwrap_or_else(|| panic!("{name} Z={z}"));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&baked), bits(&computed), "{name} Z={z}");
    }
}

#[test]
fn is_a_sad_stamp_lookup() {
    let _: SadStampLookup = builtin_sad_density;
}

#[test]
fn every_directory_names_a_compiled_in_basis() {
    // The C++ configure's refusal: a directory naming no data/basis set
    // could never be found, so it is not baked in as dead weight.
    for (basis, z, _) in BUILTIN {
        let stamp = resolve_basis(basis)
            .unwrap_or_else(|e| panic!("data/sad/{basis}/{z:03}.npy: {}", e.message));
        assert_eq!(
            stamp.name(),
            basis,
            "data/sad/{basis}/{z:03}.npy: {basis} is not the name of any data/basis set; \
             regenerate data/sad with tools/sad"
        );
    }
}

#[test]
fn every_element_of_every_compiled_in_set_is_baked() {
    for name in COMPILED_IN_SETS {
        let stamp = resolve_basis(name).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!(stamp.name(), name);
        let normalized = stamp.primitive_normalized();
        for (&z, element) in stamp.element_shells() {
            let n: usize = element
                .iter()
                .map(|shell| shell.function_count_sphe() as usize)
                .sum();
            let Some(baked) = builtin_sad_density(name, z) else {
                // The one exemption: an atom too large for the atomic SCF,
                // which the generator skips because the runtime refuses it
                // the same way (the 3d row of aug-cc-pVTZ). Anything else
                // unbaked is a stale data/sad.
                let computed = sad_density(&atom_basis(&normalized, z), &lone_atom(z), None);
                let error =
                    computed.expect_err(&format!("{name} Z={z} is not baked; regenerate data/sad"));
                assert_eq!(error.kind, ErrorKind::Unsupported, "{name} Z={z}");
                continue;
            };
            assert_eq!(baked.len(), n * n, "{name} Z={z}");
        }
    }
}

#[test]
fn nothing_else_is_baked() {
    // The canonical name, not a spelling: only a basis resolved from the
    // binary carries a name, and it carries this one.
    assert!(builtin_sad_density("cc-pvdz", 8).is_none());
    assert!(builtin_sad_density("", 8).is_none());
    // cc-pVDZ has no potassium, and nothing has element 0.
    assert!(builtin_sad_density("cc-pVDZ", 19).is_none());
    assert!(builtin_sad_density("STO-3G", 0).is_none());
}

#[test]
fn baked_matches_computed_for_hco() {
    // The one check of the stamps' values, the bitwise check a data/basis or
    // atomic-SCF change fails until data/sad is regenerated; the others
    // check only which stamps exist.
    for name in COMPILED_IN_SETS {
        expect_baked_matches_computed(name, |z| z == 1 || z == 6 || z == 8);
    }
}
