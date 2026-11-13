//! `:harmonics`: the closed-form Cartesian-to-spherical block of one
//! angular momentum.
//!
//! **The closed form.** The coefficient of `x^lx y^ly z^lz` in the real solid
//! harmonic `(l, m)` is the Schlegel-Frisch expansion (IJQC 54, 83 (1995);
//! Helgaker, "Molecular Electronic-Structure Theory", Eqs. 6.4.47-6.4.49), a
//! finite triple sum of binomials computed directly: no orthogonalization,
//! no linear solve. Columns run `m = -l ..= l`, the PySCF/libcint order.
//!
//! **G-orthonormal.** Each column is normalized under the Gaussian-moment
//! inner product `<x^a1 y^b1 z^c1, x^a2 y^b2 z^c2> = (a1+a2-1)!! (b1+b2-1)!!
//! (c1+c2-1)!!` (zero if any sum is odd), under which the harmonics are
//! orthogonal by construction, so the block `B` has `Bᵀ G B = I`. What
//! reconciles that with this tree's Cartesian normalization is one scalar
//! per shell, applied by [`crate::spherical_transform_matrix`], not here.
//!
//! Every loop is the C++ one, in the same order, so each entry is the same
//! double.

use fb_core_primitives::{binomial, double_factorial_odd, factorial};

use crate::{cartesian_powers, function_count_cart, function_count_sphe};

/// The Cartesian-to-spherical block for momentum `l >= 0`:
/// `function_count_cart(l)` × `2l+1`, column-major (`(row, col)` at
/// `row + col * ncart`), rows over [`cartesian_powers`]`(l)`, columns the
/// real solid harmonics `m = -l ..= l`: `solid_harmonic_block`.
///
/// The identity for `l <= 1`. **Unscaled**: it lacks the `sqrt((2l-1)!!)` a
/// shell's block of `U` carries, so transform a matrix with
/// [`crate::spherical_transform_matrix`], never with this.
#[must_use]
pub fn solid_harmonic_block(l: i32) -> Vec<f64> {
    let ncart = function_count_cart(l) as usize;
    let nsph = function_count_sphe(l) as usize;

    // s and p: PySCF presents these as the bare Cartesian set.
    if l <= 1 {
        let mut identity = vec![0.0; ncart * nsph];
        for i in 0..ncart {
            identity[i + i * ncart] = 1.0;
        }
        return identity;
    }

    let powers = cartesian_powers(l);
    let mut block = vec![0.0; ncart * nsph];
    for col in 0..nsph {
        // The one signed quantity: columns run m = -l ..= l.
        let m = col as i32 - l;
        add_solid_harmonic(l, m, &powers, &mut block[col * ncart..(col + 1) * ncart]);
    }

    // G-normalize each column, so blockᵀ G block = I.
    for col in block.chunks_exact_mut(ncart) {
        let mut norm2 = 0.0;
        for (a, &ca) in col.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            for (b, &cb) in col.iter().enumerate() {
                if cb == 0.0 {
                    continue;
                }
                norm2 += ca * cb * monomial_gram(powers[a], powers[b]);
            }
        }
        let inv = 1.0 / norm2.sqrt();
        for c in col {
            *c *= inv;
        }
    }
    block
}

/// The angular Gaussian-moment inner product of two same-degree monomials;
/// zero unless every per-axis sum is even (an odd moment of a centered
/// Gaussian vanishes by parity).
fn monomial_gram(p: [i32; 3], q: [i32; 3]) -> f64 {
    let (sx, sy, sz) = (p[0] + q[0], p[1] + q[1], p[2] + q[2]);
    if sx % 2 != 0 || sy % 2 != 0 || sz % 2 != 0 {
        return 0.0;
    }
    double_factorial_odd(sx - 1) * double_factorial_odd(sy - 1) * double_factorial_odd(sz - 1)
}

/// Adds the (unnormalized) real solid harmonic `(l, m)`'s coefficients into
/// `column`, one entry per monomial of `powers`: the C++
/// `solid_harmonic_coeffs`, whose map entries accumulate the same terms in
/// the same order.
///
/// The Schlegel-Frisch triple sum over `t`, `u` and a magnetic index `v`,
/// half-integer-offset for `m < 0`, so tracked as `2v`. The prefactor only
/// scales the column, which is normalized afterwards, but it is kept so the
/// bare directions match PySCF's to rounding.
fn add_solid_harmonic(l: i32, m: i32, powers: &[[i32; 3]], column: &mut [f64]) {
    let ma = m.abs();
    let prefactor = (1.0 / (2.0_f64.powi(l) * factorial(l)))
        * (2.0 * factorial(l + ma) * factorial(l - ma) / if m == 0 { 2.0 } else { 1.0 }).sqrt();

    // 2v's start: 0 for m >= 0, 1 for m < 0 (the sine branch's offset).
    let two_v0 = if m >= 0 { 0 } else { 1 };
    for t in 0..=(l - ma) / 2 {
        for u in 0..=t {
            for two_v in (two_v0..=ma).step_by(2) {
                let lx = 2 * t + ma - 2 * u - two_v;
                let ly = 2 * u + two_v;
                let lz = l - 2 * t - ma;
                if lx < 0 || ly < 0 || lz < 0 {
                    continue;
                }
                let sign_power = t + (two_v - two_v0) / 2;
                let term = if sign_power % 2 == 0 { 1.0 } else { -1.0 }
                    * 0.25_f64.powi(t)
                    * binomial(l, t)
                    * binomial(l - t, ma + t)
                    * binomial(t, u)
                    * binomial(ma, two_v);
                let row = powers.iter().position(|&p| p == [lx, ly, lz]).unwrap();
                column[row] += prefactor * term;
            }
        }
    }
}

/// `test/basis/harmonics_host.cpp`, one test per case, each named after it.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moments::{monomial_gram, row_of};
    use std::f64::consts::PI;

    /// `BasisHarmonics.SolidHarmonicsBelowDAreIdentity`.
    #[test]
    fn solid_harmonics_below_d_are_identity() {
        assert_eq!(solid_harmonic_block(0), vec![1.0]);
        assert_eq!(
            solid_harmonic_block(1),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    /// `BasisHarmonics.SolidHarmonicsAreGOrthonormal`, at its 1e-12.
    #[test]
    fn solid_harmonics_are_g_orthonormal() {
        for l in 0..=5 {
            let powers = cartesian_powers(l);
            let ncart = powers.len();
            let nsph = 2 * l as usize + 1;
            let b = solid_harmonic_block(l);
            assert_eq!(b.len(), ncart * nsph);
            for m1 in 0..nsph {
                for m2 in 0..nsph {
                    let mut dot = 0.0;
                    for p in 0..ncart {
                        for q in 0..ncart {
                            dot += b[p + m1 * ncart]
                                * monomial_gram(powers[p], powers[q])
                                * b[q + m2 * ncart];
                        }
                    }
                    let expected = if m1 == m2 { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() <= 1e-12,
                        "l = {l} m1 = {m1} m2 = {m2}: {dot}"
                    );
                }
            }
        }
    }

    /// The block for momentum `l` against libcint's (PySCF's) cart2sph
    /// block, given row by row (rows in `cartesian_powers` order, columns
    /// `m = -l ..= l`). They differ only in the overall normalization, so
    /// one positive scalar, read off the entry at (`row`, `col`), maps every
    /// column of libcint's onto ours, at 1e-14.
    fn expect_matches_libcint_up_to_one_scale(l: i32, libcint: &[f64], row: [i32; 3], col: usize) {
        let block = solid_harmonic_block(l);
        let ncart = function_count_cart(l) as usize;
        let nsph = function_count_sphe(l) as usize;
        assert_eq!(block.len(), libcint.len());
        let r0 = row_of(l, row);
        let scale = block[r0 + col * ncart] / libcint[r0 * nsph + col];
        assert!(scale > 0.0, "{scale}");
        for r in 0..ncart {
            for c in 0..nsph {
                let (ours, theirs) = (block[r + c * ncart], scale * libcint[r * nsph + c]);
                assert!(
                    (ours - theirs).abs() <= 1e-14,
                    "l = {l} {r},{c}: {ours} vs {theirs}"
                );
            }
        }
    }

    /// `BasisHarmonics.SolidHarmonicDMatchesLibcintUpToOneScale`: libcint's
    /// d block, columns `m = -2 ..= 2` (dxy, dyz, dz2, dxz, dx2-y2), in
    /// closed form.
    #[test]
    fn solid_harmonic_d_matches_libcint_up_to_one_scale() {
        let a = (15.0 / (4.0 * PI)).sqrt();
        let b = (5.0 / (16.0 * PI)).sqrt();
        let c = (15.0 / (16.0 * PI)).sqrt();
        // Rows xx, xy, xz, yy, yz, zz.
        #[rustfmt::skip]
        let libcint = [
            0.0, 0.0, -b,      0.0, c,
            a,   0.0, 0.0,     0.0, 0.0,
            0.0, 0.0, 0.0,     a,   0.0,
            0.0, 0.0, -b,      0.0, -c,
            0.0, a,   0.0,     0.0, 0.0,
            0.0, 0.0, 2.0 * b, 0.0, 0.0,
        ];
        expect_matches_libcint_up_to_one_scale(2, &libcint, [1, 1, 0], 0);
    }

    /// `BasisHarmonics.SolidHarmonicFMatchesLibcintUpToOneScale`: libcint's
    /// f block, columns `m = -3 ..= 3`, in closed form.
    #[test]
    fn solid_harmonic_f_matches_libcint_up_to_one_scale() {
        let a = (35.0 / (32.0 * PI)).sqrt();
        let b = (105.0 / (4.0 * PI)).sqrt();
        let c = (21.0 / (32.0 * PI)).sqrt();
        let d = (7.0 / (16.0 * PI)).sqrt();
        let e = (105.0 / (16.0 * PI)).sqrt();
        // Rows xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
        #[rustfmt::skip]
        let libcint = [
            0.0,     0.0, 0.0,     0.0,      -c,      0.0, a,
            3.0 * a, 0.0, -c,      0.0,      0.0,     0.0, 0.0,
            0.0,     0.0, 0.0,     -3.0 * d, 0.0,     e,   0.0,
            0.0,     0.0, 0.0,     0.0,      -c,      0.0, -3.0 * a,
            0.0,     b,   0.0,     0.0,      0.0,     0.0, 0.0,
            0.0,     0.0, 0.0,     0.0,      4.0 * c, 0.0, 0.0,
            -a,      0.0, -c,      0.0,      0.0,     0.0, 0.0,
            0.0,     0.0, 0.0,     -3.0 * d, 0.0,     -e,  0.0,
            0.0,     0.0, 4.0 * c, 0.0,      0.0,     0.0, 0.0,
            0.0,     0.0, 0.0,     2.0 * d,  0.0,     0.0, 0.0,
        ];
        expect_matches_libcint_up_to_one_scale(3, &libcint, [1, 1, 1], 1);
    }

    /// `BasisHarmonics.SolidHarmonicFFollowsRealHarmonicConvention`.
    #[test]
    fn solid_harmonic_f_follows_real_harmonic_convention() {
        let block = solid_harmonic_block(3);
        let at = |p: [i32; 3], m: i32| block[row_of(3, p) + (m + 3) as usize * 10];
        let near = |x: f64, y: f64| (x - y).abs() <= 1e-14;
        // m = -3: 3x^2 y - y^3
        assert!(at([2, 1, 0], -3) > 0.0);
        assert!(near(at([0, 3, 0], -3) / at([2, 1, 0], -3), -1.0 / 3.0));
        // m = 0: 2z^3 - 3x^2 z - 3y^2 z
        assert!(at([0, 0, 3], 0) > 0.0);
        assert!(near(at([2, 0, 1], 0) / at([0, 0, 3], 0), -1.5));
        assert!(near(at([0, 2, 1], 0) / at([0, 0, 3], 0), -1.5));
        assert_eq!(at([1, 1, 1], 0), 0.0);
        // m = +3: x^3 - 3x y^2
        assert!(at([3, 0, 0], 3) > 0.0);
        assert!(near(at([1, 2, 0], 3) / at([3, 0, 0], 3), -3.0));
    }
}
