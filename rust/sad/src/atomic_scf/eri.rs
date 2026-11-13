//! `:eri`: one atom's two-electron integrals over the Cartesian layout.
//! `atomic_integrals` calls it and carries the tensor to the spherical
//! layout.

use std::f64::consts::PI;

use fb_basis::{BasisSet, cartesian_powers};

use super::hermite::{one_center_hermite, one_center_hermite_coulomb};

/// `(ab|cd)` over `basis`'s Cartesian layout, every shell on one center:
/// `nao_cart⁴` values, `(ab|cd)` at `a + n(b + n(c + n d))`:
/// `one_center_eri`.
///
/// `(ab|cd) = 2π^{5/2} / (p q √(p+q)) Σ_{tuv} E^{ab}_{tuv} Σ_{t'u'v'}
/// (−1)^{t'+u'+v'} E^{cd}_{t'u'v'} R_{t+t',u+u',v+v'}(α)`,
/// `α = p q / (p + q)`. Only the canonical shell quartets are evaluated,
/// `sa ≥ sb`, `sc ≥ sd` and pair `(sa, sb) ≥` pair `(sc, sd)`, and each value
/// is written to all eight of `(ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) =
/// (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)`, so the tensor is exactly
/// symmetric. Each quartet's value is the C++ sum, its primitives and its
/// Hermite indices in the C++ order.
pub(super) fn one_center_eri(basis: &BasisSet) -> Vec<f64> {
    let ncart = basis.nao_cart() as usize;
    let shells = basis.shells();
    let nsh = shells.len();
    let powers: Vec<Vec<[i32; 3]>> = shells
        .iter()
        .map(|shell| cartesian_powers(shell.angular_momentum()))
        .collect();

    let mut eri = vec![0.0; ncart * ncart * ncart * ncart];
    let at =
        |a: usize, b: usize, c: usize, d: usize| a + (ncart * (b + (ncart * (c + (ncart * d)))));
    let prefactor = 2.0 * PI.powf(2.5);
    let mut block: Vec<f64> = Vec::new();
    for sa in 0..nsh {
        for sb in 0..=sa {
            for sc in 0..=sa {
                // Pair (sc, sd) may not pass pair (sa, sb): at sc == sa, sd
                // stops at sb.
                let sd_last = if sc == sa { sb } else { sc };
                for sd in 0..=sd_last {
                    let off = [sa, sb, sc, sd].map(|s| basis.ao_offset_cart(s as u32) as usize);
                    let len = [sa, sb, sc, sd].map(|s| powers[s].len());
                    let lab = shells[sa].angular_momentum() + shells[sb].angular_momentum();
                    let lcd = shells[sc].angular_momentum() + shells[sd].angular_momentum();
                    // The quartet's values, contracted over every primitive
                    // before they are scattered: i + na(j + nb(k + nc l)).
                    block.clear();
                    block.resize(len[0] * len[1] * len[2] * len[3], 0.0);
                    for pa in shells[sa].primitives() {
                        for pb in shells[sb].primitives() {
                            let p = pa.exponent + pb.exponent;
                            let ep = one_center_hermite(lab, p);
                            for pc in shells[sc].primitives() {
                                for pd in shells[sd].primitives() {
                                    let q = pc.exponent + pd.exponent;
                                    let eq = one_center_hermite(lcd, q);
                                    let alpha = p * q / (p + q);
                                    let scale = pa.coefficient
                                        * pb.coefficient
                                        * pc.coefficient
                                        * pd.coefficient
                                        * prefactor
                                        / (p * q * (p + q).sqrt());
                                    for i in 0..len[0] {
                                        for j in 0..len[1] {
                                            let a = powers[sa][i];
                                            let b = powers[sb][j];
                                            let n = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
                                            for k in 0..len[2] {
                                                for l in 0..len[3] {
                                                    let c = powers[sc][k];
                                                    let d = powers[sd][l];
                                                    let m = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                                                    // Odd total power on any axis
                                                    // integrates to zero.
                                                    if (n[0] + m[0]) % 2 != 0
                                                        || (n[1] + m[1]) % 2 != 0
                                                        || (n[2] + m[2]) % 2 != 0
                                                    {
                                                        continue;
                                                    }
                                                    let sum = hermite_sum(&ep, n, &eq, m, alpha);
                                                    block[i
                                                        + (len[0]
                                                            * (j + (len[1]
                                                                * (k + (len[2] * l)))))] +=
                                                        scale * sum;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Assigned, not accumulated: every element of the tensor
                    // lies in exactly one canonical quartet's orbit, and a
                    // quartet with a repeated shell or pair writes some
                    // elements more than once, always with the same value.
                    for l in 0..len[3] {
                        for k in 0..len[2] {
                            for j in 0..len[1] {
                                for i in 0..len[0] {
                                    let v =
                                        block[i + (len[0] * (j + (len[1] * (k + (len[2] * l)))))];
                                    let (a, b, c, d) =
                                        (off[0] + i, off[1] + j, off[2] + k, off[3] + l);
                                    eri[at(a, b, c, d)] = v;
                                    eri[at(b, a, c, d)] = v;
                                    eri[at(a, b, d, c)] = v;
                                    eri[at(b, a, d, c)] = v;
                                    eri[at(c, d, a, b)] = v;
                                    eri[at(d, c, a, b)] = v;
                                    eri[at(c, d, b, a)] = v;
                                    eri[at(d, c, b, a)] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eri
}

/// The Hermite double sum of one component quartet and one primitive
/// quartet, the bra's powers `n` and the ket's `m`: the C++ loop body's six
/// nested loops, in its order and with its grouping.
fn hermite_sum(ep: &[Vec<f64>], n: [i32; 3], eq: &[Vec<f64>], m: [i32; 3], alpha: f64) -> f64 {
    let [nx, ny, nz] = n;
    let [mx, my, mz] = m;
    let mut sum = 0.0;
    for tx in (nx % 2..=nx).step_by(2) {
        for ty in (ny % 2..=ny).step_by(2) {
            for tz in (nz % 2..=nz).step_by(2) {
                let eab = ep[nx as usize][tx as usize]
                    * ep[ny as usize][ty as usize]
                    * ep[nz as usize][tz as usize];
                for ux in (mx % 2..=mx).step_by(2) {
                    for uy in (my % 2..=my).step_by(2) {
                        for uz in (mz % 2..=mz).step_by(2) {
                            let sign = if (ux + uy + uz) % 2 == 0 { 1.0 } else { -1.0 };
                            sum += eab
                                * sign
                                * eq[mx as usize][ux as usize]
                                * eq[my as usize][uy as usize]
                                * eq[mz as usize][uz as usize]
                                * one_center_hermite_coulomb(tx + ux, ty + uy, tz + uz, alpha);
                        }
                    }
                }
            }
        }
    }
    sum
}
