//! `:one_electron`: one atom's one-electron matrices over the Cartesian
//! layout, S, T and the attraction V to its own nucleus. `atomic_integrals`
//! calls these and carries their results to the spherical layout.
//!
//! The loop order (shell pair, primitive pair, component pair) and each
//! integrand's grouping are the C++ ones, so every element is the same sum
//! of the same terms in the same order.

use std::f64::consts::PI;

use fb_basis::{BasisSet, cartesian_powers};

use super::hermite::{one_center_hermite, one_center_hermite_coulomb};

/// The loop the three matrices share, `one_center_pair_loop`: every shell
/// pair, every primitive pair, every Cartesian component pair, accumulating
/// `cc · integrand(a, b, p, β, e)` into `(a, b)`. The Hermite table runs to
/// `la + lb + extra_n`.
fn one_center_pair_loop(
    basis: &BasisSet,
    extra_n: i32,
    integrand: impl Fn([i32; 3], [i32; 3], f64, f64, &[Vec<f64>]) -> f64,
) -> Vec<f64> {
    let ncart = basis.nao_cart() as usize;
    let shells = basis.shells();
    let mut out = vec![0.0; ncart * ncart];
    for (sa, shell_a) in (0u32..).zip(shells) {
        let powers_a = cartesian_powers(shell_a.angular_momentum());
        let oa = basis.ao_offset_cart(sa) as usize;
        for (sb, shell_b) in (0u32..).zip(shells) {
            let powers_b = cartesian_powers(shell_b.angular_momentum());
            let ob = basis.ao_offset_cart(sb) as usize;
            let lab = shell_a.angular_momentum() + shell_b.angular_momentum();
            for pa in shell_a.primitives() {
                for pb in shell_b.primitives() {
                    let p = pa.exponent + pb.exponent;
                    let cc = pa.coefficient * pb.coefficient;
                    let e = one_center_hermite(lab + extra_n, p);
                    for (i, &a) in powers_a.iter().enumerate() {
                        for (j, &b) in powers_b.iter().enumerate() {
                            out[(oa + i) + (ncart * (ob + j))] +=
                                cc * integrand(a, b, p, pb.exponent, &e);
                        }
                    }
                }
            }
        }
    }
    out
}

/// The one-dimensional overlap of `x^n exp(−p x²)`, its Hermite `t = 0`
/// term: `one_center_overlap_1d`.
fn one_center_overlap_1d(e: &[Vec<f64>], n: i32, p: f64) -> f64 {
    if n < 0 {
        0.0
    } else {
        e[n as usize][0] * (PI / p).sqrt()
    }
}

/// S over `basis`'s Cartesian layout, `nao_cart × nao_cart`, column-major:
/// `one_center_overlap`. Every shell of `basis` is taken to sit on one
/// center.
pub(super) fn one_center_overlap(basis: &BasisSet) -> Vec<f64> {
    one_center_pair_loop(basis, 0, |a, b, p, _, e| {
        one_center_overlap_1d(e, a[0] + b[0], p)
            * one_center_overlap_1d(e, a[1] + b[1], p)
            * one_center_overlap_1d(e, a[2] + b[2], p)
    })
}

/// `T = −½ ⟨a|∇²|b⟩`, as [`one_center_overlap`]: `one_center_kinetic`.
pub(super) fn one_center_kinetic(basis: &BasisSet) -> Vec<f64> {
    // +2 for the x^{n+2} the Laplacian makes.
    one_center_pair_loop(basis, 2, |a, b, p, beta, e| {
        let mut s = [0.0; 3];
        let mut t = [0.0; 3];
        for x in 0..3 {
            let n = a[x] + b[x];
            s[x] = one_center_overlap_1d(e, n, p);
            // d²/dx² on the ket's x^b exp(−β x²), then integrated.
            let bx = f64::from(b[x]);
            t[x] = (bx * (bx - 1.0) * one_center_overlap_1d(e, n - 2, p))
                - (2.0 * beta * ((2.0 * bx) + 1.0) * one_center_overlap_1d(e, n, p))
                + (4.0 * beta * beta * one_center_overlap_1d(e, n + 2, p));
        }
        -0.5 * ((t[0] * s[1] * s[2]) + (s[0] * t[1] * s[2]) + (s[0] * s[1] * t[2]))
    })
}

/// V, the attraction to a nucleus of charge `z` at the shells' common
/// center, as [`one_center_overlap`]: `one_center_nuclear`.
pub(super) fn one_center_nuclear(z: i32, basis: &BasisSet) -> Vec<f64> {
    // V = −z (2π / p) Σ_{tuv} E_t E_u E_v R_{tuv}(p), the nucleus at the
    // product's center.
    let charge = f64::from(z);
    one_center_pair_loop(basis, 0, move |a, b, p, _, e| {
        let nx = a[0] + b[0];
        let ny = a[1] + b[1];
        let nz = a[2] + b[2];
        let mut sum = 0.0;
        for tx in (nx % 2..=nx).step_by(2) {
            for ty in (ny % 2..=ny).step_by(2) {
                for tz in (nz % 2..=nz).step_by(2) {
                    sum += e[nx as usize][tx as usize]
                        * e[ny as usize][ty as usize]
                        * e[nz as usize][tz as usize]
                        * one_center_hermite_coulomb(tx, ty, tz, p);
                }
            }
        }
        -charge * 2.0 * PI / p * sum
    })
}
