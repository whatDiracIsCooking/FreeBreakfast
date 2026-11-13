//! `:integrals`: one atom's integrals over the spherical layout of its
//! shells, [`AtomicIntegrals`] and [`atomic_integrals`].

use fb_basis::{BasisSet, Shell, spherical_transform_matrix};
use fb_core_error_handling::{Error, Result};

use super::eri::one_center_eri;
use super::one_electron::{one_center_kinetic, one_center_nuclear, one_center_overlap};

/// The most Cartesian AOs an atom may have, `kMaxAtomicAoCart`: the n⁴
/// two-electron tensor is held whole, over the Cartesian layout first, and
/// 96⁴ doubles is 680 MB. An atom in a triple-zeta basis is well inside it.
pub(super) const MAX_ATOMIC_AO_CART: usize = 96;

/// The rank-agnostic change of one index, `rotate_transform`: contracts the
/// slowest index of `input` (extent `u_rows`) with the `u_rows × u_cols`
/// column-major `u` and makes the result the fastest index. Applied once per
/// index, it carries a rank-r tensor over the Cartesian layout to the
/// spherical one with every index back in place. The C++ loop order, so the
/// same sums.
fn rotate_transform(input: &[f64], u: &[f64], u_rows: usize, u_cols: usize) -> Vec<f64> {
    let rest = input.len() / u_rows;
    let mut out = vec![0.0; rest * u_cols];
    for d in 0..u_rows {
        for l in 0..u_cols {
            let w = u[d + (u_rows * l)];
            if w == 0.0 {
                continue;
            }
            for r in 0..rest {
                out[l + (u_cols * r)] += w * input[r + (rest * d)];
            }
        }
    }
    out
}

/// `m`, of rank `rank` over the Cartesian layout, carried to the spherical
/// one: `to_spherical`.
fn to_spherical(mut m: Vec<f64>, u: &[f64], ncart: usize, nsph: usize, rank: i32) -> Vec<f64> {
    for _ in 0..rank {
        m = rotate_transform(&m, u, ncart, nsph);
    }
    m
}

/// One atom's integrals over the spherical layout of its shells,
/// `AtomicIntegrals`: `n` = the sum of each shell's `function_count_sphe()`,
/// every matrix flat and column-major.
pub(crate) struct AtomicIntegrals {
    /// n, the spherical AO count.
    pub(crate) nao_sphe: usize,
    /// S, n × n.
    pub(crate) overlap: Vec<f64>,
    /// T, n × n.
    pub(crate) kinetic: Vec<f64>,
    /// V, n × n: attraction to the atom's own nucleus, charge z.
    pub(crate) nuclear: Vec<f64>,
    /// `(ij|kl)` at `i + n(j + n(k + n l))`, n⁴.
    pub(crate) eri: Vec<f64>,
}

/// S, T, V and the two-electron integrals of one atom's `shells`, over
/// their spherical layout: `atomic_integrals`.
///
/// Computed over the Cartesian components and brought to the spherical
/// layout with [`spherical_transform_matrix`] of the shells, so the
/// convention is the tree's: coefficients multiply unnormalized Cartesian
/// primitives, and the spherical columns run `m = −l..+l`. `z` is the
/// nuclear charge V attracts to; the shells are all on one center (which
/// one does not matter: the integrals are translation invariant).
///
/// `InvalidConfig` if `shells` is empty or its centers differ;
/// `Unsupported` past 96 Cartesian AOs.
pub(crate) fn atomic_integrals(z: i32, shells: &[Shell]) -> Result<AtomicIntegrals> {
    let Some(front) = shells.first() else {
        return Err(Error::config("atomic_integrals: the atom has no shells"));
    };
    let center = front.center();
    for shell in shells {
        let c = shell.center();
        if c.x != center.x || c.y != center.y || c.z != center.z {
            return Err(Error::config(
                "atomic_integrals: the shells are not all on one center",
            ));
        }
    }
    let basis = BasisSet::new(shells.to_vec());
    let ncart = basis.nao_cart() as usize;
    let nsph = basis.nao_sphe() as usize;
    if ncart > MAX_ATOMIC_AO_CART {
        return Err(Error::unsupported(format!(
            "atomic_integrals: {ncart} Cartesian AOs, past the {MAX_ATOMIC_AO_CART} the n^4 \
             tensor is held for"
        )));
    }

    let u = spherical_transform_matrix(&basis);
    Ok(AtomicIntegrals {
        nao_sphe: nsph,
        overlap: to_spherical(one_center_overlap(&basis), &u, ncart, nsph, 2),
        kinetic: to_spherical(one_center_kinetic(&basis), &u, ncart, nsph, 2),
        nuclear: to_spherical(one_center_nuclear(z, &basis), &u, ncart, nsph, 2),
        eri: to_spherical(one_center_eri(&basis), &u, ncart, nsph, 4),
    })
}
