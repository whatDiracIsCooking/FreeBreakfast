//! `:basis_set`: an ordered collection of shells with its two AO layouts,
//! and the Cartesian-to-spherical transform built from it.

use fb_core_error_handling::{Error, check_or_abort};
use fb_core_primitives::double_factorial_odd;

use crate::{Ao, Shell, cartesian_powers, solid_harmonic_block};

/// An ordered collection of shells with atomic-orbital indexing:
/// `BasisSet`.
///
/// **Two parallel AO layouts.** The AOs are each shell's functions
/// concatenated in shell order: shell 0 owns `[0, n0)`, shell 1
/// `[n0, n0 + n1)`, and so on. In the Cartesian layout the kernels evaluate
/// ([`BasisSet::nao_cart`], [`BasisSet::ao_offset_cart`],
/// [`BasisSet::aos_cart`]) `n` is `(l+1)(l+2)/2`; in the spherical layout
/// the basis means ([`BasisSet::nao_sphe`], [`BasisSet::ao_offset_sphe`])
/// it is `2l+1`. [`spherical_transform_matrix`] bridges the two.
///
/// A basis made by [`crate::BasisStamp::place`] also records which atom owns
/// which shells, so per-atom work reads the grouping rather than re-deriving
/// it from centers.
///
/// Construction aborts if the Cartesian AO count passes `i32::MAX`, so every
/// count and offset fits the `int32_t` the kernels' AO offsets and 32-bit
/// BLAS dimensions take. `Default` is the empty basis.
#[derive(Clone, Debug, Default)]
pub struct BasisSet {
    shells: Vec<Shell>,
    // One offset per shell plus a trailing total, in each layout; empty
    // only for the default basis.
    ao_offsets_cart: Vec<u32>,
    ao_offsets_sphe: Vec<u32>,
    // One AO per Cartesian function, in global AO order.
    aos_cart: Vec<Ao>,
    // One offset per atom plus a trailing total; empty with no per-atom map.
    atom_shell_offsets: Vec<u32>,
    name: String,
}

impl BasisSet {
    /// The basis over `shells`, whose order fixes the AO layout, with no
    /// per-atom shell map ([`BasisSet::natoms`] is 0) and no name.
    #[track_caller]
    pub fn new(shells: Vec<Shell>) -> BasisSet {
        BasisSet::build(shells, Vec::new(), String::new())
    }

    /// The basis over `shells`, grouped by atom: atom `a` owns shells
    /// `[atom_shell_offsets[a], atom_shell_offsets[a + 1])`, so the offsets
    /// must start at 0, never decrease and end at `shells.len()`; anything
    /// else aborts at the caller, since the one producer,
    /// [`crate::BasisStamp::place`], builds them in the loop that builds
    /// `shells`. `name` is the compiled-in set the shells were placed from,
    /// or empty ([`BasisSet::name`]).
    #[track_caller]
    pub fn with_atoms(
        shells: Vec<Shell>,
        atom_shell_offsets: Vec<u32>,
        name: impl Into<String>,
    ) -> BasisSet {
        let valid = atom_shell_offsets.first() == Some(&0)
            && atom_shell_offsets.last().map(|&n| n as usize) == Some(shells.len())
            && atom_shell_offsets.is_sorted();
        if !valid {
            check_or_abort::<()>(Err(Error::config(
                "BasisSet: atom shell offsets do not partition the shells",
            )));
        }
        BasisSet::build(shells, atom_shell_offsets, name.into())
    }

    /// Lays out both AO layouts over `shells`, aborting past `i32::MAX`
    /// Cartesian AOs.
    #[track_caller]
    fn build(shells: Vec<Shell>, atom_shell_offsets: Vec<u32>, name: String) -> BasisSet {
        let mut ao_offsets_cart = Vec::with_capacity(shells.len() + 1);
        let mut ao_offsets_sphe = Vec::with_capacity(shells.len() + 1);
        let mut aos_cart = Vec::new();
        let (mut offset, mut spherical_offset) = (0u32, 0u32);
        for (s, shell) in (0u32..).zip(&shells) {
            ao_offsets_cart.push(offset);
            ao_offsets_sphe.push(spherical_offset);
            for [a, b, c] in cartesian_powers(shell.angular_momentum()) {
                aos_cart.push(Ao {
                    shell_index: s,
                    a: a as u8,
                    b: b as u8,
                    c: c as u8,
                });
            }
            offset += shell.function_count_cart();
            spherical_offset += shell.function_count_sphe();
            // Per shell, so a u32 wrap cannot hide an oversized basis: one
            // shell adds far less than 2^31.
            if offset > i32::MAX as u32 {
                check_or_abort::<()>(Err(Error::config(format!(
                    "BasisSet: more than {} Cartesian AOs does not fit a 32-bit AO offset",
                    i32::MAX
                ))));
            }
        }
        ao_offsets_cart.push(offset);
        ao_offsets_sphe.push(spherical_offset);
        BasisSet {
            shells,
            ao_offsets_cart,
            ao_offsets_sphe,
            aos_cart,
            atom_shell_offsets,
            name,
        }
    }

    /// The number of atoms the shells are grouped by: the molecule's for a
    /// placed basis, 0 for one built from a bare shell list.
    pub fn natoms(&self) -> u32 {
        self.atom_shell_offsets.len().saturating_sub(1) as u32
    }

    /// The first shell atom `atom` owns. An atom not below
    /// [`BasisSet::natoms`] aborts at the caller.
    #[track_caller]
    pub fn atom_shell_offset(&self, atom: u32) -> u32 {
        if atom >= self.natoms() {
            check_or_abort::<()>(Err(Error::config(
                "BasisSet::atom_shell_offset: atom index out of range",
            )));
        }
        self.atom_shell_offsets[atom as usize]
    }

    /// The number of shells atom `atom` owns. An atom not below
    /// [`BasisSet::natoms`] aborts at the caller.
    #[track_caller]
    pub fn atom_shell_count(&self, atom: u32) -> u32 {
        if atom >= self.natoms() {
            check_or_abort::<()>(Err(Error::config(
                "BasisSet::atom_shell_count: atom index out of range",
            )));
        }
        self.atom_shell_offsets[atom as usize + 1] - self.atom_shell_offsets[atom as usize]
    }

    /// The canonical name of the compiled-in set the shells were placed
    /// from ([`crate::BasisStamp::name`]), or empty: set only where the
    /// shells are known to be that set's, since the SAD guess looks its
    /// baked densities up by it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The shells, in the order supplied.
    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }

    /// The Cartesian AOs in global AO order, [`BasisSet::nao_cart`] of them:
    /// each shell's [`cartesian_powers`] in shell order.
    pub fn aos_cart(&self) -> &[Ao] {
        &self.aos_cart
    }

    /// The number of shells.
    pub fn nshells(&self) -> u32 {
        self.shells.len() as u32
    }

    /// The number of AOs in the Cartesian layout.
    pub fn nao_cart(&self) -> u32 {
        self.ao_offsets_cart.last().copied().unwrap_or(0)
    }

    /// The first Cartesian AO shell `shell_index` owns. An index not below
    /// [`BasisSet::nshells`] aborts at the caller, as `std::vector::at`
    /// would throw: every caller loops over the basis's own shells.
    #[track_caller]
    pub fn ao_offset_cart(&self, shell_index: u32) -> u32 {
        if shell_index >= self.nshells() {
            check_or_abort::<()>(Err(Error::config(
                "BasisSet::ao_offset_cart: shell index out of range",
            )));
        }
        self.ao_offsets_cart[shell_index as usize]
    }

    /// The number of AOs in the spherical layout; [`BasisSet::nao_cart`]
    /// exactly when no shell is above p.
    pub fn nao_sphe(&self) -> u32 {
        self.ao_offsets_sphe.last().copied().unwrap_or(0)
    }

    /// The first spherical AO shell `shell_index` owns. An index not below
    /// [`BasisSet::nshells`] aborts at the caller.
    #[track_caller]
    pub fn ao_offset_sphe(&self, shell_index: u32) -> u32 {
        if shell_index >= self.nshells() {
            check_or_abort::<()>(Err(Error::config(
                "BasisSet::ao_offset_sphe: shell index out of range",
            )));
        }
        self.ao_offsets_sphe[shell_index as usize]
    }
}

/// The Cartesian-to-spherical transform `U` for `basis`: `nao_cart` ×
/// `nao_sphe`, column-major (`(i, j)` at `i + j * nao_cart`, ready for a
/// BLAS call), so that `Uᵀ M U` turns a Cartesian-AO matrix `M` into the same
/// quantity over the spherical layout: `spherical_transform_matrix`.
///
/// Block-diagonal, one block per shell at
/// ([`BasisSet::ao_offset_cart`], [`BasisSet::ao_offset_sphe`]): the shell's
/// [`solid_harmonic_block`] scaled by `sqrt((2l-1)!!)`.
///
/// **Why that scale, and nothing else.** Every Cartesian component of a shell
/// shares one radial contraction, so the harmonic block combines them
/// whatever that radial part's normalization. The block is orthonormal under
/// the raw monomial moments, while this tree's Cartesian AOs carry
/// [`crate::ShellStamp::primitive_normalized`]'s `(l,0,0)` scale on every
/// component; `(2l-1)!!` is exactly the ratio of the two. So for one
/// normalized primitive, `Uᵀ S U` is the identity on the shell's block, and
/// for a contraction (never renormalized) a uniform scalar times it, every
/// cross-`m` entry zero either way.
///
/// `U`'s columns are orthonormal under the Cartesian overlap, not the plain
/// dot product: `Uᵀ U` is not the identity.
#[must_use]
pub fn spherical_transform_matrix(basis: &BasisSet) -> Vec<f64> {
    let nrow = basis.nao_cart() as usize;
    let ncol = basis.nao_sphe() as usize;
    let mut u = vec![0.0; nrow * ncol];
    for (s, shell) in (0u32..).zip(basis.shells()) {
        let l = shell.angular_momentum();
        let row0 = basis.ao_offset_cart(s) as usize;
        let col0 = basis.ao_offset_sphe(s) as usize;
        let ncart = shell.function_count_cart() as usize;
        let nsph = shell.function_count_sphe() as usize;
        let block = solid_harmonic_block(l);
        let scale = double_factorial_odd(2 * l - 1).sqrt();
        for i in 0..ncart {
            for j in 0..nsph {
                u[row0 + i + (col0 + j) * nrow] = block[i + j * ncart] * scale;
            }
        }
    }
    u
}

/// `test/basis/basis_set_host.cpp`, one test per case, each named after it,
/// and the acceptance check on a basis through f. The abort cases are
/// `tests/basis_abort.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moments::{monomial_gram, row_of};
    use crate::{Primitive, ShellStamp};
    use fb_core_types::Double3;
    use std::f64::consts::PI;

    const ORIGIN: Double3 = Double3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// A one-primitive shell of momentum `l` at the origin.
    fn shell(l: i32, exponent: f64, coefficient: f64) -> Shell {
        Shell::new(
            l,
            vec![Primitive {
                exponent,
                coefficient,
            }],
            ORIGIN,
        )
    }

    /// `BasisSet.EmptyBasisHasNoFunctions`.
    #[test]
    fn empty_basis_has_no_functions() {
        let basis = BasisSet::default();
        assert_eq!(basis.nshells(), 0);
        assert_eq!(basis.nao_cart(), 0);
        assert_eq!(basis.nao_sphe(), 0);
        assert!(basis.aos_cart().is_empty());
        assert!(spherical_transform_matrix(&basis).is_empty());
        // An empty shell list is the same basis.
        let built = BasisSet::new(Vec::new());
        assert_eq!(built.nao_cart(), 0);
        assert_eq!(built.nao_sphe(), 0);
    }

    /// `BasisSet.MovedFromBasisHasNoFunctions`: what a move leaves in Rust
    /// is nothing to read, so the state checked is what `mem::take` leaves.
    #[test]
    fn moved_from_basis_has_no_functions() {
        let mut basis = BasisSet::new(vec![shell(0, 1.0, 1.0), shell(2, 1.0, 1.0)]);
        let taken = std::mem::take(&mut basis);
        assert_eq!(taken.nao_cart(), 7);
        assert_eq!(taken.nao_sphe(), 6);
        assert_eq!(basis.nao_cart(), 0);
        assert_eq!(basis.nao_sphe(), 0);
    }

    /// `BasisSet.AtomShellOffsetsGroupShells`.
    #[test]
    fn atom_shell_offsets_group_shells() {
        assert_eq!(BasisSet::new(vec![shell(0, 1.0, 1.0)]).natoms(), 0);

        // Atom 0: s + p, atom 1: no shells, atom 2: s.
        let basis = BasisSet::with_atoms(
            vec![shell(0, 1.0, 1.0), shell(1, 1.0, 1.0), shell(0, 1.0, 1.0)],
            vec![0, 2, 2, 3],
            "",
        );
        assert_eq!(basis.natoms(), 3);
        assert_eq!(basis.atom_shell_offset(0), 0);
        assert_eq!(basis.atom_shell_count(0), 2);
        assert_eq!(basis.atom_shell_count(1), 0);
        assert_eq!(basis.atom_shell_offset(2), 2);
        assert_eq!(basis.atom_shell_count(2), 1);
    }

    /// `BasisSet.NameIsWhatTheConstructorWasGiven`.
    #[test]
    fn name_is_what_the_constructor_was_given() {
        assert!(BasisSet::new(vec![shell(0, 1.0, 1.0)]).name().is_empty());
        assert!(
            BasisSet::with_atoms(vec![shell(0, 1.0, 1.0)], vec![0, 1], "")
                .name()
                .is_empty()
        );
        let named = BasisSet::with_atoms(vec![shell(0, 1.0, 1.0)], vec![0, 1], "sto-3g");
        assert_eq!(named.name(), "sto-3g");
        assert_eq!(named.natoms(), 1);
    }

    /// `BasisSet.LayoutsConcatenateShellsInOrder`.
    #[test]
    fn layouts_concatenate_shells_in_order() {
        // s, p, d.
        let basis = BasisSet::new(vec![
            shell(0, 3.42, 0.15),
            shell(1, 0.62, 1.0),
            shell(2, 0.8, 1.0),
        ]);
        assert_eq!(basis.nshells(), 3);
        assert_eq!(basis.nao_cart(), 10); // 1 + 3 + 6: the d shell's Cartesian functions
        assert_eq!(basis.nao_sphe(), 9); // 1 + 3 + 5
        assert_eq!(basis.ao_offset_cart(0), 0);
        assert_eq!(basis.ao_offset_cart(1), 1);
        assert_eq!(basis.ao_offset_cart(2), 4);
        assert_eq!(basis.ao_offset_sphe(2), 4);

        let aos_cart = basis.aos_cart();
        assert_eq!(aos_cart.len(), basis.nao_cart() as usize);
        for s in 0..basis.nshells() {
            let powers = cartesian_powers(basis.shells()[s as usize].angular_momentum());
            for (i, p) in powers.iter().enumerate() {
                let ao = aos_cart[basis.ao_offset_cart(s) as usize + i];
                assert_eq!(ao.shell_index, s);
                assert_eq!([ao.a, ao.b, ao.c].map(i32::from), *p);
            }
        }
    }

    /// The acceptance check: for a basis with every momentum through f, the
    /// counts, offsets and transform shape are the C++ values exactly, as a
    /// program over `basis.h`'s own `function_count_cart` /
    /// `function_count_sphe` printed them (cart 1, 3, 6, 10 at 0, 1, 4, 10;
    /// spherical 1, 3, 5, 7 at 0, 1, 4, 9; 20 and 16 in all).
    #[test]
    fn layouts_through_f_are_the_cpp_integers() {
        let basis = BasisSet::new(
            (0..=crate::MAX_ANGULAR_MOMENTUM)
                .map(|l| shell(l, 1.0, 1.0))
                .collect(),
        );
        let counts: Vec<[u32; 2]> = basis
            .shells()
            .iter()
            .map(|s| [s.function_count_cart(), s.function_count_sphe()])
            .collect();
        assert_eq!(counts, [[1, 1], [3, 3], [6, 5], [10, 7]]);
        let offsets: Vec<[u32; 2]> = (0..basis.nshells())
            .map(|s| [basis.ao_offset_cart(s), basis.ao_offset_sphe(s)])
            .collect();
        assert_eq!(offsets, [[0, 0], [1, 1], [4, 4], [10, 9]]);
        assert_eq!(basis.nao_cart(), 20);
        assert_eq!(basis.nao_sphe(), 16);
        assert_eq!(basis.aos_cart().len(), 20);
        assert_eq!(spherical_transform_matrix(&basis).len(), 20 * 16);
    }

    /// `BasisSet.SphericalTransformMatrixIsIdentityBelowD`.
    #[test]
    fn spherical_transform_matrix_is_identity_below_d() {
        // s and p are the same functions in either convention.
        let basis = BasisSet::new(vec![
            shell(0, 1.0, 1.0),
            shell(1, 1.0, 1.0),
            shell(0, 0.5, 1.0),
        ]);
        let n = basis.nao_cart() as usize;
        assert_eq!(basis.nao_sphe() as usize, n);
        let u = spherical_transform_matrix(&basis);
        assert_eq!(u.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(u[i + j * n], if i == j { 1.0 } else { 0.0 }, "{i},{j}");
            }
        }
    }

    /// `BasisSet.SphericalShellOverlapTransformsToIdentity`, at its 1e-12:
    /// for one normalized primitive at one center the Cartesian overlap is
    /// `S_pq = c² (π/2α)^{3/2} G(p, q) / (4α)^l` in closed form, and `Uᵀ S U`
    /// must be the identity on the spherical block.
    #[test]
    fn spherical_shell_overlap_transforms_to_identity() {
        for l in 2..=4 {
            let alpha = 1.3;
            let stamp = ShellStamp::new(
                l,
                vec![Primitive {
                    exponent: alpha,
                    coefficient: 1.0,
                }],
            );
            let basis = BasisSet::new(vec![stamp.primitive_normalized().place(ORIGIN)]);
            let c = basis.shells()[0].primitives()[0].coefficient;
            let powers = cartesian_powers(l);
            let ncart = basis.nao_cart() as usize;
            let nsph = basis.nao_sphe() as usize;
            assert_eq!(nsph, 2 * l as usize + 1);

            let u = spherical_transform_matrix(&basis);
            for m1 in 0..nsph {
                for m2 in 0..nsph {
                    let mut s = 0.0;
                    for p in 0..ncart {
                        for q in 0..ncart {
                            let s_pq = c
                                * c
                                * (PI / (2.0 * alpha)).powf(1.5)
                                * monomial_gram(powers[p], powers[q])
                                / (4.0 * alpha).powi(l);
                            s += u[p + m1 * ncart] * s_pq * u[q + m2 * ncart];
                        }
                    }
                    let expected = if m1 == m2 { 1.0 } else { 0.0 };
                    assert!(
                        (s - expected).abs() <= 1e-12,
                        "l = {l} m1 = {m1} m2 = {m2}: {s}"
                    );
                }
            }
        }
    }

    /// `BasisSet.SphericalTransformMatrixPlacesBlocksAtOffsets`, at its
    /// 1e-15: s then d, a 7 × 6 block-diagonal `U`.
    #[test]
    fn spherical_transform_matrix_places_blocks_at_offsets() {
        let basis = BasisSet::new(vec![shell(0, 1.0, 1.0), shell(2, 1.0, 1.0)]);
        let ncol = basis.nao_sphe() as usize;
        assert_eq!(basis.nao_cart(), 7);
        assert_eq!(ncol, 6);
        let u = spherical_transform_matrix(&basis);
        let d = solid_harmonic_block(2);
        assert_eq!(u[0], 1.0);
        for j in 1..ncol {
            assert_eq!(u[j * 7], 0.0); // s row, d columns
        }
        for i in 1..7 {
            assert_eq!(u[i], 0.0); // d rows, s column
            for j in 0..5 {
                let expected = 3.0_f64.sqrt() * d[(i - 1) + j * 6];
                assert!((u[i + (1 + j) * 7] - expected).abs() <= 1e-15, "{i},{j}");
            }
        }
    }

    /// `BasisSet.SphericalTransformMatrixIsNotOrthonormal`: the d_xy column
    /// (`m = -2`) is `sqrt(3)` in the xy row and zero elsewhere, so its
    /// `Uᵀ U` diagonal is 3.
    #[test]
    fn spherical_transform_matrix_is_not_orthonormal() {
        let basis = BasisSet::new(vec![shell(2, 1.0, 1.0)]);
        let ncart = basis.nao_cart() as usize;
        let nsph = basis.nao_sphe() as usize;
        let u = spherical_transform_matrix(&basis);
        assert_eq!(u.len(), ncart * nsph);

        let mut utu = vec![0.0; nsph * nsph];
        for a in 0..nsph {
            for b in 0..nsph {
                for i in 0..ncart {
                    utu[a + b * nsph] += u[i + a * ncart] * u[i + b * ncart];
                }
            }
        }
        assert!((u[row_of(2, [1, 1, 0])] - 3.0_f64.sqrt()).abs() <= 1e-15);
        assert!((utu[0] - 3.0).abs() <= 1e-14);
        let mut off_identity = 0.0_f64;
        for a in 0..nsph {
            for b in 0..nsph {
                let id = if a == b { 1.0 } else { 0.0 };
                off_identity = off_identity.max((utu[a + b * nsph] - id).abs());
            }
        }
        assert!(off_identity > 1.0, "{off_identity}");
    }
}
