//! `src/basis` (module `free_breakfast.basis`): the description of a
//! Gaussian basis, the data an integral engine consumes. It evaluates no
//! integrals.
//!
//! Everything is in atomic units: shell centers in bohr, primitive exponents
//! in inverse bohr².
//!
//! **Spherical basis, evaluated as Cartesian.** Every shell means the `2l+1`
//! real solid harmonics of its angular momentum `l`, and is evaluated over
//! the `(l+1)(l+2)/2` Cartesian Gaussians `x^a y^b z^c`, `a + b + c = l`, in
//! the one canonical order [`cartesian_powers`] gives (`a` highest, then `b`):
//! `(x, y, z)` for p, `(xx, xy, xz, yy, yz, zz)` for d. A [`BasisSet`]
//! therefore has two parallel AO layouts, the Cartesian one the kernels
//! evaluate and the spherical one the basis means, and
//! [`spherical_transform_matrix`] bridges them.
//!
//! **Contractions are not renormalized.** A shell's coefficients are stored
//! as supplied; [`ShellStamp::primitive_normalized`] folds in each
//! primitive's own normalization and nothing more, as in the C++ tree.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `angular_momentum.h`: `kMaxAngularMomentum` | [`MAX_ANGULAR_MOMENTUM`], asserted equal to the header's value |
//! | `basis.h`: `function_count_cart`, `function_count_sphe` | [`function_count_cart`], [`function_count_sphe`], `const fn` |
//! | `basis.h`: `Primitive` | [`Primitive`] |
//! | `:shell`: `Shell(l, primitives, center)` | [`Shell::new`], aborting on the same inputs |
//! | `:shell`: `Shell(l, exponents, coefficients, center)` | [`Shell::zipped`] |
//! | `:shell`: `angular_momentum`, `primitives`, `nprimitives`, `center`, `function_count_cart`, `function_count_sphe` | the [`Shell`] methods of those names |
//! | `:ao`: `cartesian_powers` | [`cartesian_powers`] |
//! | `:ao`: `AO` | [`Ao`] |
//! | `:harmonics`: `solid_harmonic_block` | [`solid_harmonic_block`] |
//! | `:basis_set`: `BasisSet()`, `BasisSet(shells)` | [`BasisSet::default`], [`BasisSet::new`] |
//! | `:basis_set`: `BasisSet(shells, atom_shell_offsets, name = {})` | [`BasisSet::with_atoms`], the name always given |
//! | `:basis_set`: `natoms`, `atom_shell_offset`, `atom_shell_count`, `name`, `shells`, `aos_cart`, `nshells`, `nao_cart`, `ao_offset_cart`, `nao_sphe`, `ao_offset_sphe` | the [`BasisSet`] methods of those names |
//! | `:basis_set`: `spherical_transform_matrix` | [`spherical_transform_matrix`] |
//! | `:shell_stamp`: `ShellStamp`, both constructors, its accessors and `primitive_normalized` | [`ShellStamp::new`], [`ShellStamp::zipped`] and the methods of those names |
//! | `:shell_stamp`: `place(stamp, center)` | [`ShellStamp::place`] |
//! | `:basis_stamp`: `BasisStamp`, `name`, `element_shells`, `primitive_normalized` | [`BasisStamp`] and the methods of those names |
//! | `:basis_stamp`: `shells_for`, a pointer or `nullptr` | [`BasisStamp::shells_for`], `Option<&[ShellStamp]>` |
//! | `:basis_stamp`: `place(stamp, molecule)` | [`BasisStamp::place`] |
//! | the `free_breakfast::basis::headers` include path, and the re-exports from the global module fragment | nothing: one crate, one definition |
//! | the re-export of `free_breakfast.molecule` | nothing: a dependent names `fb_molecule` itself |
//!
//! **The two `place` overloads are methods.** Rust has no overloading, and
//! the receiver says which: `shell_stamp.place(center)` makes a [`Shell`],
//! `basis_stamp.place(&molecule)` a [`BasisSet`].
//!
//! **There is no moved-from basis.** The C++ counts check for the empty
//! offset vectors a move leaves; a Rust move leaves nothing to read. The
//! empty state that remains is [`BasisSet::default`], which `std::mem::take`
//! leaves behind, and it reads as no shells and no functions.
//!
//! **The angular-momentum ceiling is the crate's.** In the C++ tree
//! `kMaxAngularMomentum` lives beside `function_count_cart`, in a header the
//! basis module and device code share, so it is here too, not in
//! `fb_core_constants`. Its test reads the C++ header and checks the two
//! agree. Nothing here enforces it: a shell of any `l` is accepted, and an
//! engine refuses what its kernels do not cover.
//!
//! **Placing onto a molecule needs no device.** A C++ `Molecule` owns its
//! nuclei on the device, so `place(stamp, molecule)` is tested in a GPU
//! binary there; an `fb_molecule::Molecule` is host data, so those cases are
//! unit tests here.

#![forbid(unsafe_code)]

mod ao;
mod basis_set;
mod basis_stamp;
mod harmonics;
mod shell;
mod shell_stamp;

pub use ao::{Ao, cartesian_powers};
pub use basis_set::{BasisSet, spherical_transform_matrix};
pub use basis_stamp::BasisStamp;
pub use harmonics::solid_harmonic_block;
pub use shell::{MAX_ANGULAR_MOMENTUM, Primitive, Shell, function_count_cart, function_count_sphe};
pub use shell_stamp::ShellStamp;

/// `test/basis/gaussian_moments.h`: the closed-form Gaussian-moment algebra
/// the harmonics and basis-set cases check against, written out
/// independently of the crate's own.
#[cfg(test)]
mod moments {
    /// `(n)!!` over odd `n`, the empty product for `n <= 1`.
    pub fn double_factorial(n: i32) -> f64 {
        let mut result = 1.0;
        let mut k = n;
        while k > 1 {
            result *= f64::from(k);
            k -= 2;
        }
        result
    }

    /// The angular Gaussian-moment inner product the harmonic table is
    /// normalized under: the product over axes of `(p + q - 1)!!`, zero if
    /// any axis sum is odd.
    pub fn monomial_gram(p: [i32; 3], q: [i32; 3]) -> f64 {
        let mut result = 1.0;
        for axis in 0..3 {
            let s = p[axis] + q[axis];
            if s % 2 != 0 {
                return 0.0;
            }
            result *= double_factorial(s - 1);
        }
        result
    }

    /// The row of the Cartesian monomial `p` within its shell of momentum
    /// `l`.
    pub fn row_of(l: i32, p: [i32; 3]) -> usize {
        super::cartesian_powers(l)
            .iter()
            .position(|&q| q == p)
            .unwrap()
    }
}
