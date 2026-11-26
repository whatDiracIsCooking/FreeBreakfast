//! The Cartesian Schwarz diagonal of a placed [`BasisSet`]: `sqrt((mn|mn))`
//! for every Cartesian function `m` of a shell `a` and `n` of a shell `b`,
//! over the shell pairs a caller lists, on the host, at planning.
//!
//! **Why the host, at planning.** The density-fitting screen drops a shell
//! pair whose every element is below `schwarz_threshold`, and the kept
//! pairs fix the plan's sizes: the columns of the packed B, its heap row,
//! the batches that stream it. Admission then sizes the plan from them
//! (`docs/memory.md`, "Admission"), and planning runs before any task can,
//! so the numbers it reads must exist before the plan does: this is a
//! deliberate host-only numerical path, not a CPU twin of a device task. It
//! costs `O(shell pairs · K⁴)` for `K` primitives a shell, once per plan.
//!
//! **What it computes.** With the Coulomb metric,
//! `Σ_Q B_Q,mn² = (mn|mn)_DF <= (mn|mn)`, the fit being a projection, so a
//! dropped pair's columns of B have 2-norm below the threshold. Each
//! element is
//!
//! ```text
//! (mn|mn) = Σ_{ij} Σ_{kl} c_i c_j c_k c_l 2π^{5/2} / (p q √(p+q))
//!           Σ_{tuv} E^{ij}_{tuv} Σ_{τνφ} (-1)^{τ+ν+φ} E^{kl}_{τνφ} R_{t+τ,u+ν,v+φ}(pq/(p+q), P - Q)
//! ```
//!
//! by McMurchie–Davidson, general in `l` (Helgaker, Jørgensen and Olsen,
//! "Molecular Electronic-Structure Theory", Eq. 9.9.33): `ij` and `kl` run
//! over the pair's primitive pairs, bra and ket alike, `p` and `q` their
//! exponent sums, the Hermite coefficients `E` by Eqs. 9.5.6 and 9.5.7 from
//! `E^{00}_0 = exp(-μ X_AB²)` per axis, and the Hermite integrals `R` by
//! Eqs. 9.9.18 to 9.9.20 from the Boys function through order
//! `4 l_max = 12` at f (the host DF metric's
//! [`fb_integrals_df_metric::boys`]). Only the diagonal is formed,
//! `ncart_a · ncart_b` per pair, each primitive quartet and its swap once
//! (they are the same integral), and nothing is transformed to the
//! spherical harmonics.
//!
//! **The normalisation is the device kernels'.** A Cartesian function is
//! `x^a y^b z^c` times the shell's contraction, its coefficients as the
//! shell holds them on every component
//! ([`fb_basis::ShellStamp::primitive_normalized`]'s scale, a contraction
//! never renormalised), in [`fb_basis::cartesian_powers`]' order: the
//! functions the records and every integral kernel evaluate. The libcint
//! comparison pins it (`tests/libcint.rs`).
//!
//! **The order is the caller's.** The packed B's columns are the shell
//! pairs in `fb_integrals_mmd::record_tables`' class order, but that
//! function is a crate whose normal dependencies reach the kernel
//! library, and this one's reach none (only its tier-4 test's do), so the
//! caller lists the pairs, from the tables'
//! `ClassPairs` (`bra[p]`, `ket[p]`), and the diagonal comes back in that
//! order: one definition of the order, the tables'.
//!
//! No C++ counterpart: the C++ tree screens on the device's per-record
//! bounds, which need the records first.
//!
//! [`BasisSet`]: fb_basis::BasisSet

#![forbid(unsafe_code)]

use std::f64::consts::PI;

use contracts::debug_requires;
use fb_basis::{BasisSet, MAX_ANGULAR_MOMENTUM, Shell, cartesian_powers};
use fb_integrals_df_metric::boys;

/// The Cartesian Schwarz diagonal of `basis` over `pairs`: for each pair
/// `(a, b)` of shell indices, in the order given, the block
/// `sqrt((mn|mn))`, `m` over shell `a`'s Cartesian functions and `n` over
/// shell `b`'s, `m` slower (element `(m, n)` at `m · ncart_b + n` of the
/// block), the blocks one after another. So the result holds
/// `Σ ncart_a · ncart_b` doubles, and a pair's largest element is its
/// screen's bound.
///
/// ```
/// use fb_basis::{BasisSet, Shell};
///
/// // An s and a p shell on one centre, the origin.
/// let basis = BasisSet::new(vec![
///     Shell::zipped(0, &[1.0], &[0.712_705_470_354_990_2], Default::default()),
///     Shell::zipped(1, &[0.5], &[0.423_777_208_123_757_7], Default::default()),
/// ]);
/// let q = fb_integrals_schwarz::schwarz_diagonal(&basis, &[(0, 0), (0, 1), (1, 1)]);
/// assert_eq!(q.len(), 1 + 3 + 9);
/// // A unit s Gaussian of exponent α: (ss|ss) = 2 √(α/π) = 2/√π here.
/// assert!((q[0] - (2.0 / std::f64::consts::PI.sqrt()).sqrt()).abs() < 1e-14);
/// ```
#[debug_requires(
    pairs.iter().all(|&(a, b)| a < basis.nshells() && b < basis.nshells()),
    "every pair's shells are the basis's"
)]
#[debug_requires(
    basis.shells().iter().all(|s| s.angular_momentum() <= MAX_ANGULAR_MOMENTUM),
    "every shell within f, the momenta the Boys function's accuracy is fixed at"
)]
pub fn schwarz_diagonal(basis: &BasisSet, pairs: &[(u32, u32)]) -> Vec<f64> {
    let shells = basis.shells();
    let mut out = Vec::new();
    for &(a, b) in pairs {
        pair_diagonal(&shells[a as usize], &shells[b as usize], &mut out);
    }
    out
}

/// One primitive pair of a shell pair: its exponent sum, centre, the
/// product of its two coefficients, and its Hermite coefficients per axis,
/// `e[axis][(i (lb + 1) + j) (la + lb + 1) + t] = E^{ij}_t`.
struct PrimitivePair {
    p: f64,
    centre: [f64; 3],
    coefficient: f64,
    e: [Vec<f64>; 3],
}

/// The pair's Cartesian `sqrt((mn|mn))`, `m` of `sa` slower, appended to
/// `out`.
fn pair_diagonal(sa: &Shell, sb: &Shell, out: &mut Vec<f64>) {
    let (la, lb) = (
        sa.angular_momentum() as usize,
        sb.angular_momentum() as usize,
    );
    let (ca, cb) = (sa.center(), sb.center());
    let (ra, rb) = ([ca.x, ca.y, ca.z], [cb.x, cb.y, cb.z]);
    let nt = la + lb + 1;

    let mut prims = Vec::new();
    for pa in sa.primitives() {
        for pb in sb.primitives() {
            let (alpha, beta) = (pa.exponent, pb.exponent);
            let p = alpha + beta;
            let mu = alpha * beta / p;
            let centre: [f64; 3] = std::array::from_fn(|x| (alpha * ra[x] + beta * rb[x]) / p);
            let e = std::array::from_fn(|x| {
                hermite(
                    la,
                    lb,
                    p,
                    centre[x] - ra[x],
                    centre[x] - rb[x],
                    (-mu * (ra[x] - rb[x]).powi(2)).exp(),
                )
            });
            prims.push(PrimitivePair {
                p,
                centre,
                coefficient: pa.coefficient * pb.coefficient,
                e,
            });
        }
    }

    let powers_a = cartesian_powers(la as i32);
    let powers_b = cartesian_powers(lb as i32);
    let l_tot = 2 * (la + lb);
    let ns = l_tot + 1;
    let mut sum = vec![0.0; powers_a.len() * powers_b.len()];
    let mut r = vec![0.0; ns * ns * ns];
    let mut scratch = vec![0.0; ns * ns * ns];
    let mut f = vec![0.0; ns];
    // G[axis][(i (lb + 1) + j) ns + s] = Σ_{t+τ=s} E^{ij}_t (-1)^τ E'^{ij}_τ.
    let mut g: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; (la + 1) * (lb + 1) * ns]);

    for (k1, bra) in prims.iter().enumerate() {
        for (k2, ket) in prims.iter().enumerate().skip(k1) {
            let (p, q) = (bra.p, ket.p);
            let alpha = p * q / (p + q);
            let pq: [f64; 3] = std::array::from_fn(|x| bra.centre[x] - ket.centre[x]);
            let t = alpha * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
            boys(t, &mut f);
            hermite_integrals(l_tot, alpha, pq, &f, &mut r, &mut scratch);

            for (x, gx) in g.iter_mut().enumerate() {
                gx.fill(0.0);
                for ij in 0..(la + 1) * (lb + 1) {
                    let (e1, e2) = (&bra.e[x][ij * nt..][..nt], &ket.e[x][ij * nt..][..nt]);
                    let gij = &mut gx[ij * ns..][..ns];
                    for (t1, &v1) in e1.iter().enumerate() {
                        for (t2, &v2) in e2.iter().enumerate() {
                            let sign = if t2 % 2 == 0 { 1.0 } else { -1.0 };
                            gij[t1 + t2] += v1 * sign * v2;
                        }
                    }
                }
            }

            // The quartet and its swap are the same integral.
            let twice = if k1 == k2 { 1.0 } else { 2.0 };
            let pre = twice * 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt())
                * bra.coefficient
                * ket.coefficient;
            let mut slot = 0;
            for pa in &powers_a {
                for pb in &powers_b {
                    let at = |x: usize| {
                        let (i, j) = (pa[x] as usize, pb[x] as usize);
                        (&g[x][(i * (lb + 1) + j) * ns..][..ns], i + j)
                    };
                    let ((gx, nx), (gy, ny), (gz, nz)) = (at(0), at(1), at(2));
                    let mut v = 0.0;
                    for (s0, &g0) in gx.iter().enumerate().take(2 * nx + 1) {
                        for (s1, &g1) in gy.iter().enumerate().take(2 * ny + 1) {
                            let row = &r[(s0 * ns + s1) * ns..][..2 * nz + 1];
                            let mut w = 0.0;
                            for (&g2, &rv) in gz.iter().zip(row) {
                                w += g2 * rv;
                            }
                            v += g0 * g1 * w;
                        }
                    }
                    sum[slot] += pre * v;
                    slot += 1;
                }
            }
        }
    }
    // (mn|mn) is a norm squared; a rounding below zero is zero.
    out.extend(sum.iter().map(|v| v.max(0.0).sqrt()));
}

/// The Hermite coefficients `E^{ij}_t` of one axis, `i <= la`, `j <= lb`,
/// `t <= i + j`, at `(i (lb + 1) + j) (la + lb + 1) + t`: from
/// `E^{00}_0 = k`, `E^{i+1,j}_t = E^{ij}_{t-1} / 2p + X_PA E^{ij}_t
/// + (t+1) E^{ij}_{t+1}`, and its twin in `j` with `X_PB`.
fn hermite(la: usize, lb: usize, p: f64, xpa: f64, xpb: f64, k: f64) -> Vec<f64> {
    let nt = la + lb + 1;
    let at = |i: usize, j: usize| (i * (lb + 1) + j) * nt;
    let mut e = vec![0.0; (la + 1) * (lb + 1) * nt];
    e[0] = k;
    let half_p = 0.5 / p;
    // From (i, j) to the next index along one side, `x` its distance.
    let step = |e: &mut Vec<f64>, from: usize, to: usize, n: usize, x: f64| {
        for t in 0..=n + 1 {
            let lower = if t > 0 { e[from + t - 1] } else { 0.0 };
            let upper = if t < n {
                (t + 1) as f64 * e[from + t + 1]
            } else {
                0.0
            };
            let here = if t <= n { e[from + t] } else { 0.0 };
            e[to + t] = half_p * lower + x * here + upper;
        }
    };
    for i in 0..=la {
        if i > 0 {
            step(&mut e, at(i - 1, 0), at(i, 0), i - 1, xpa);
        }
        for j in 1..=lb {
            step(&mut e, at(i, j - 1), at(i, j), i + j - 1, xpb);
        }
    }
    e
}

/// The Hermite integrals `R_{tuv} = R^{(0)}_{tuv}` for `t + u + v <= l`,
/// into `r` at `(t (l + 1) + u) (l + 1) + v`, from the Boys values `f`:
/// `R^{(n)}_{000} = (-2α)^n F_n`, then each order from the one above,
/// `R^{(n)}_{t+1,u,v} = t R^{(n+1)}_{t-1,u,v} + X_PQ R^{(n+1)}_{tuv}` and its
/// twins in `u` and `v`, `scratch` the order above.
fn hermite_integrals(
    l: usize,
    alpha: f64,
    pq: [f64; 3],
    f: &[f64],
    r: &mut [f64],
    scratch: &mut [f64],
) {
    let n1 = l + 1;
    let at = |t: usize, u: usize, v: usize| (t * n1 + u) * n1 + v;
    // Each order swaps the two first, so the last, order 0, lands in `r`.
    let (mut cur, mut prev) = if l.is_multiple_of(2) {
        (scratch, r)
    } else {
        (r, scratch)
    };
    for n in (0..=l).rev() {
        std::mem::swap(&mut cur, &mut prev);
        let m = l - n;
        cur[0] = (-2.0 * alpha).powi(n as i32) * f[n];
        for t in 0..=m {
            for u in 0..=m - t {
                for v in 0..=m - t - u {
                    let (value, from) = if t > 0 {
                        (pq[0], [t - 1, u, v])
                    } else if u > 0 {
                        (pq[1], [t, u - 1, v])
                    } else if v > 0 {
                        (pq[2], [t, u, v - 1])
                    } else {
                        continue;
                    };
                    let axis = usize::from(t == 0) + usize::from(t == 0 && u == 0);
                    let k = from[axis];
                    let mut x = value * prev[at(from[0], from[1], from[2])];
                    if k > 0 {
                        let mut back = from;
                        back[axis] -= 1;
                        x += k as f64 * prev[at(back[0], back[1], back[2])];
                    }
                    cur[at(t, u, v)] = x;
                }
            }
        }
    }
}
