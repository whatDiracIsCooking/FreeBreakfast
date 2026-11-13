//! `:ao`: the canonical Cartesian order and one Cartesian atomic orbital.

use crate::function_count_cart;

/// The Cartesian powers `(a, b, c)` of a shell of momentum `l`, in the
/// tree's canonical order, `a` highest, then `b`: `cartesian_powers`.
///
/// s → `(0,0,0)`; p → `(x, y, z)`; d → `(xx, xy, xz, yy, yz, zz)`. The result
/// has [`function_count_cart`]`(l)` entries, and the order is load-bearing:
/// the kernels' Cartesian tiles and the per-AO normalization are laid out
/// against it. Defined for any `l >= 0`.
#[must_use]
pub fn cartesian_powers(l: i32) -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(function_count_cart(l) as usize);
    for a in (0..=l).rev() {
        for b in (0..=l - a).rev() {
            out.push([a, b, l - a - b]);
        }
    }
    out
}

/// One Cartesian atomic orbital: the shell that owns it and its powers
/// `(a, b, c)`, `a + b + c = l`: `AO`.
///
/// It stores no center or exponents; `shell_index` resolves those through
/// the owning [`crate::BasisSet`], and its place in
/// [`crate::BasisSet::aos_cart`] is its global AO index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ao {
    /// The owning shell's index in the basis.
    pub shell_index: u32,
    /// The power of `x`.
    pub a: u8,
    /// The power of `y`.
    pub b: u8,
    /// The power of `z`.
    pub c: u8,
}

/// `test/basis/ao_host.cpp`'s ordering case; its counts case is the shell
/// module's.
#[cfg(test)]
mod tests {
    use super::*;

    /// `BasisAo.CartesianPowersCanonicalOrder`.
    #[test]
    fn cartesian_powers_canonical_order() {
        assert_eq!(cartesian_powers(0), vec![[0, 0, 0]]);
        assert_eq!(cartesian_powers(1), vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        // xx, xy, xz, yy, yz, zz
        assert_eq!(
            cartesian_powers(2),
            vec![
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]
            ]
        );
        // Defined past d, with every entry of degree l.
        for l in 3..=5 {
            let powers = cartesian_powers(l);
            assert_eq!(powers.len(), function_count_cart(l) as usize);
            assert_eq!(powers[0], [l, 0, 0]);
            assert_eq!(*powers.last().unwrap(), [0, 0, l]);
            for p in &powers {
                assert_eq!(p[0] + p[1] + p[2], l);
            }
        }
    }
}
