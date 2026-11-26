//! The density-fitting Coulomb metric `M[P][Q] = (P|Q)`, the two-center
//! two-electron integral over a fitting set, computed on the host: plain
//! Rust on rayon's threads, no device and no kernel library.
//!
//! **The integral.** For two primitive Cartesian Gaussians of exponents `a`
//! and `b` on centers `A` and `B`, McMurchie-Davidson with one function per
//! electron (Helgaker, "Molecular Electronic-Structure Theory", 9.9):
//!
//! ```text
//! (a|b) = 2 pi^(5/2) / (a b sqrt(a + b))
//!         * (-1)^lb * sum_{tuv} sum_{t'u'v'} E^a_tuv E^b_t'u'v' R_{t+t',u+u',v+v'}
//! ```
//!
//! - `E` is the one-center Hermite expansion of a monomial,
//!   `x^i exp(-a x²) = sum_t E^i_t Λ_t` ([`hermite_expansion`]). It is zero
//!   unless `i - t` is even, which is why the sign of the second function's
//!   Hermite derivatives is `(-1)^lb` for the whole shell.
//! - `R_tuv` are the Hermite Coulomb integrals at `alpha = a b / (a + b)`
//!   over `A - B`, from the Boys function `F_n(alpha |A - B|²)`
//!   ([`boys`]) by the usual recurrence.
//!
//! **The layout.** `M` is over the `2l+1` real solid harmonics of each
//! shell, through the basis's own Cartesian-to-spherical block
//! ([`fb_basis::solid_harmonic_block`], scaled by `sqrt((2l-1)!!)` as
//! [`fb_basis::spherical_transform_matrix`] scales it). Coefficients are
//! taken as given: place the set `primitive_normalized()`, as the DF steps
//! do, and the result is the matrix
//! `rust/integrals/eri/density_fitting/metric` builds on a card.
//!
//! **The lower triangle alone.** `M` is symmetric, and [`metric`] returns
//! it packed: `(P, Q)`, `Q <= P`, at [`lower_index`], `n (n + 1) / 2`
//! values. Only the shell pairs `A >= B` are computed, one rayon task per
//! center, whose rows are one contiguous slice of the triangle
//! (the pool it is called in sets the thread count: a step's
//! [`HostPool`], or rayon's global one by `RAYON_NUM_THREADS`).
//!
//! **Directly spherical.** [`metric`] never forms a Cartesian integral. A
//! column of that block is a harmonic polynomial `H` of degree `l`, and
//! Hobson's theorem gives `H(r) exp(-a r²) = H(∇_A) exp(-a r²) / (2a)^l`:
//! a spherical Gaussian is the Hermite Gaussians of the top order alone,
//! under `H`'s coefficients. So `E` above drops out, the exponents enter
//! only through a scalar weight, a contracted shell pair sums
//! `R_tuv`, `t + u + v = la + lb`, over its primitive pairs, and an
//! element of the block is a sum over the two harmonics' nonzero Cartesian
//! terms of a lookup into that sum.
//! [`metric_via_cartesian`] is the route through `E` and the Cartesian
//! integrals, kept to check and time against.
//!
//! **The far field.** Once `x = alpha |A - B|²` is large, `exp(-x)` is
//! nothing beside the Boys values, `F_n(x) = (2n-1)!! / (2x)^n *
//! sqrt(pi / x) / 2`, and `R_tuv` is `sqrt(pi / alpha) / 2` times the
//! `tuv` derivative of `1 / |A - B|`: the exponents are left in a scalar
//! alone, and that scalar factors over the two shells. So for a pair of
//! centers the block of each `(la, lb)` is built once, from one table of
//! those derivatives, and every far pair of shells on them is that block
//! times the product of two per-shell strengths: no Boys function, no
//! recurrence and no harmonic sum of its own. A pair is far when
//! `alpha |A - B|² >= 40 + 4 (la + lb)` for its smallest `alpha`, where
//! what is dropped is under 1e-17 of what is kept; the test is per pair
//! of shells, since a tight pair is far at a distance a diffuse one is
//! not.
//!
//! The Boys tables reach order 20, so shells through `l = 10`; the
//! carried sets reach i.

#![forbid(unsafe_code)]

// The hot loops are `f64::mul_add`, which is one instruction with the FMA
// target feature and a software call, about 50% slower than a multiply and
// an add, without it. The workspace's `.cargo/config.toml` turns the feature
// on for every member; this refuses a build that lost it (cargo run from
// outside `rust/`, or a RUSTFLAGS in the environment, which replaces the
// file's). rustdoc generates no code and reads RUSTDOCFLAGS instead, so a
// documentation build is exempt.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "fma"),
    not(doc),
    not(doctest)
))]
compile_error!(
    "fb-integrals-df-metric needs the FMA target feature: build from rust/, whose \
     .cargo/config.toml sets it, or add `-C target-feature=+fma` to RUSTFLAGS"
);

mod boys_series;
mod pool;

use contracts::debug_requires;

use std::f64::consts::PI;

use fb_basis::{BasisSet, Shell, cartesian_powers, solid_harmonic_block};
use fb_core_primitives::{double_factorial_odd, lower_index};
use rayon::prelude::*;

pub use pool::HostPool;

/// The highest order [`boys`] reaches: two shells of `l = 10`.
pub const BOYS_MAX_ORDER: usize = boys_series::MAX_ORDER;

/// From here on `erf(sqrt(x))` rounds to 1, and [`boys`] does not call it.
const ERF_IS_ONE_FROM: f64 = 36.0;

/// The Boys function `F_n(x) = ∫₀¹ t^(2n) exp(-x t²) dt` for every
/// `n < f.len()`, written into `f`: the C++ tree's `eval_boys`
/// (`src/integrals/shared/eval_boys.h`), with its coefficients.
///
/// With `M` the highest order asked for: from `x = M + 8` on,
/// `F_0 = sqrt(pi / x) erf(sqrt(x)) / 2` (the `erf` skipped from `x = 36`,
/// where it is 1) and the upward recurrence
/// `F_(n+1) = ((2n+1) F_n - exp(-x)) / (2x)`. Below,
/// `F_M = exp(-x) S_M(2x)`, `S_M` the Kummer series as a polynomial of
/// fixed degree by Horner (`boys_series.rs`), and the downward recurrence
/// `F_n = (2x F_(n+1) + exp(-x)) / (2n+1)`.
///
/// Horner is `mul_add`, the header's `fma`, so the series is the
/// header's operation for operation.
#[debug_requires(!f.is_empty(), "an order is asked for")]
#[debug_requires(f.len() <= BOYS_MAX_ORDER + 1, "no order is past BOYS_MAX_ORDER")]
#[debug_requires(x >= 0.0 && x.is_finite(), "x is finite and not negative")]
pub fn boys(x: f64, f: &mut [f64]) {
    let top = f.len() - 1;
    let e = (-x).exp();
    if x >= (top + 8) as f64 {
        let h = 0.5 / x;
        // erfc(6) is 2e-17: from x = 36 on erf(sqrt(x)) is 1 in a double.
        let erf = if x >= ERF_IS_ONE_FROM {
            1.0
        } else {
            libm::erf(x.sqrt())
        };
        f[0] = 0.5 * (PI / x).sqrt() * erf;
        for n in 1..=top {
            f[n] = ((2 * n - 1) as f64 * f[n - 1] - e) * h;
        }
        return;
    }
    let argument = 2.0 * x;
    let series = boys_series::SERIES[top]
        .iter()
        .fold(0.0, |s: f64, &c| s.mul_add(argument, c));
    f[top] = e * series;
    for n in (0..top).rev() {
        f[n] = (argument * f[n + 1] + e) / (2 * n + 1) as f64;
    }
}

/// The one-center Hermite expansion coefficients `E^i_t` of
/// `x^i exp(-a x²) = sum_t E^i_t Λ_t`, `Λ_t` the `t`-th derivative of the
/// Gaussian in its center, for `i, t <= l`: `(l+1)²` values, `E^i_t` at
/// `i * (l + 1) + t`.
///
/// `E^0_0 = 1` and `E^(i+1)_t = E^i_(t-1) / (2a) + (t + 1) E^i_(t+1)`; an
/// entry with `i - t` odd, or `t > i`, is zero.
#[must_use]
pub fn hermite_expansion(l: usize, a: f64) -> Vec<f64> {
    let d = l + 1;
    let half = 0.5 / a;
    let mut e = vec![0.0; d * d];
    e[0] = 1.0;
    for i in 0..l {
        for t in 0..=i + 1 {
            let below = if t > 0 { e[i * d + t - 1] } else { 0.0 };
            let above = if t < i { e[i * d + t + 1] } else { 0.0 };
            e[(i + 1) * d + t] = half * below + (t + 1) as f64 * above;
        }
    }
    e
}

/// The Hermite Coulomb integrals `R^n_tuv` of total order up to `l` at
/// `alpha` over the separation `r`, built from
/// `R^n_000 = (-2 alpha)^n F_n(alpha r²)` by
/// `R^n_(t+1)uv = t R^(n+1)_(t-1)uv + x R^(n+1)_tuv` and its `y` and `z`
/// twins.
///
/// The table's side is the last fill's `l + 1`, not the largest it can
/// hold, so a low-order pair stays in a few cache lines.
struct HermiteCoulomb {
    /// The side of the last fill, its `l + 1`.
    d: usize,
    /// `R^n_tuv` at `((n * d + t) * d + u) * d + v`.
    r: Vec<f64>,
    /// The Boys values of one separation.
    f: Vec<f64>,
}

impl HermiteCoulomb {
    /// A table that holds every order up to `l`.
    fn new(l: usize) -> HermiteCoulomb {
        let d = l + 1;
        HermiteCoulomb {
            d,
            r: vec![0.0; d * d * d * d],
            f: vec![0.0; d],
        }
    }

    /// Fill `R^n_tuv` for `n + t + u + v <= l`, `l` at most the table's.
    fn fill(&mut self, l: usize, alpha: f64, r: [f64; 3]) {
        let d = l + 1;
        self.d = d;
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        boys(alpha * r2, &mut self.f[..=l]);
        let mut power = 1.0;
        for n in 0..=l {
            self.r[n * d * d * d] = power * self.f[n];
            power *= -2.0 * alpha;
        }
        self.recur(l, r);
    }

    /// Fill the table with the derivatives of `1 / |r|`, the far field
    /// every pair of exponents shares: `T^n_000 = (-1)^n (2n-1)!! /
    /// |r|^(2n+1)` and the same recurrence, so `T^0_tuv` is the `tuv`
    /// derivative of `1 / |r|` in `r`.
    ///
    /// It is what [`HermiteCoulomb::fill`] tends to, over
    /// `sqrt(pi / alpha) / 2`, once `exp(-alpha r²)` is nothing: there
    /// `F_n(x) = (2n-1)!! / (2x)^n * sqrt(pi / x) / 2`.
    fn fill_far(&mut self, l: usize, r: [f64; 3]) {
        let d = l + 1;
        self.d = d;
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let inverse = 1.0 / r2;
        let mut value = inverse.sqrt();
        for n in 0..=l {
            self.r[n * d * d * d] = value;
            value *= -((2 * n + 1) as f64) * inverse;
        }
        self.recur(l, r);
    }

    /// The recurrence up from the seeds `R^n_000`, `n <= l`, at side
    /// `l + 1`.
    fn recur(&mut self, l: usize, r: [f64; 3]) {
        let d = l + 1;
        let at = |n: usize, t: usize, u: usize, v: usize| ((n * d + t) * d + u) * d + v;
        let stride = d * d * d;
        for order in 1..=l {
            for t in 0..=order {
                for u in 0..=order - t {
                    let v = order - t - u;
                    // Lower the first index that is not zero: `one` below
                    // it, and `two` below it when there is one.
                    let (x, count, one, two) = if t > 0 {
                        (r[0], t - 1, at(1, t - 1, u, v), at(1, t.max(2) - 2, u, v))
                    } else if u > 0 {
                        (r[1], u - 1, at(1, t, u - 1, v), at(1, t, u.max(2) - 2, v))
                    } else {
                        (r[2], v - 1, at(1, t, u, v - 1), at(1, t, u, v.max(2) - 2))
                    };
                    let here = at(0, t, u, v);
                    let factor = count as f64;
                    for n in 0..=l - order {
                        let offset = n * stride;
                        self.r[here + offset] =
                            factor.mul_add(self.r[two + offset], x * self.r[one + offset]);
                    }
                }
            }
        }
    }

    /// `R^0_tuv`.
    fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.r[(t * self.d + u) * self.d + v]
    }
}

/// One shell as the integral loops read it: its momentum, its Cartesian
/// powers, and its Cartesian-to-spherical block with the shell's scale
/// folded in.
struct ShellView {
    /// The angular momentum.
    l: usize,
    /// The Cartesian powers, in the basis's order.
    powers: Vec<[usize; 3]>,
    /// `ncart × nsph`, column-major: [`solid_harmonic_block`] times
    /// `sqrt((2l-1)!!)`.
    harmonics: Vec<f64>,
    /// The same block by its nonzeros, the columns one after another: of a
    /// Cartesian function `x^t y^u z^v`, `t`, `u` and its coefficient.
    terms: Vec<(usize, usize, f64)>,
    /// Where each column's terms start in `terms`, and one past the last.
    term_offsets: Vec<usize>,
}

impl ShellView {
    /// The view of momentum `l`.
    fn new(l: i32) -> ShellView {
        let scale = double_factorial_odd(2 * l - 1).sqrt();
        let powers: Vec<[usize; 3]> = cartesian_powers(l)
            .iter()
            .map(|p| p.map(|c| c as usize))
            .collect();
        let harmonics: Vec<f64> = solid_harmonic_block(l).iter().map(|c| c * scale).collect();
        let mut terms = Vec::new();
        let mut term_offsets = vec![0];
        for column in harmonics.chunks_exact(powers.len()) {
            for (&[t, u, _], &c) in powers.iter().zip(column) {
                if c != 0.0 {
                    terms.push((t, u, c));
                }
            }
            term_offsets.push(terms.len());
        }
        ShellView {
            l: l as usize,
            powers,
            harmonics,
            terms,
            term_offsets,
        }
    }

    /// The nonzero terms of spherical function `m`'s column.
    fn column(&self, m: usize) -> &[(usize, usize, f64)] {
        &self.terms[self.term_offsets[m]..self.term_offsets[m + 1]]
    }

    /// The Cartesian function count.
    fn ncart(&self) -> usize {
        self.powers.len()
    }

    /// The spherical function count.
    fn nsph(&self) -> usize {
        2 * self.l + 1
    }
}

/// The contracted Cartesian block `(a|b)` of two shells, `ncart_a` rows by
/// `ncart_b` columns at `ia * ncart_b + ib`, the coefficients as given.
fn cartesian_block(
    a: &Shell,
    va: &ShellView,
    b: &Shell,
    vb: &ShellView,
    table: &mut HermiteCoulomb,
) -> Vec<f64> {
    let (ca, cb) = (a.center(), b.center());
    let r = [ca.x - cb.x, ca.y - cb.y, ca.z - cb.z];
    let (da, db) = (va.l + 1, vb.l + 1);
    let sign = if vb.l.is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut block = vec![0.0; va.ncart() * vb.ncart()];
    for pa in a.primitives() {
        let ea = hermite_expansion(va.l, pa.exponent);
        for pb in b.primitives() {
            let eb = hermite_expansion(vb.l, pb.exponent);
            let (ax, bx) = (pa.exponent, pb.exponent);
            table.fill(va.l + vb.l, ax * bx / (ax + bx), r);
            let scale = sign * TWO_PI_POW_5_2 / (ax * bx * (ax + bx).sqrt())
                * pa.coefficient
                * pb.coefficient;
            for (ia, &[i, j, k]) in va.powers.iter().enumerate() {
                for (ib, &[l, m, n]) in vb.powers.iter().enumerate() {
                    // E^i_t is zero unless i - t is even: step by two.
                    let mut sum = 0.0;
                    for t in (i % 2..=i).step_by(2) {
                        for u in (j % 2..=j).step_by(2) {
                            for v in (k % 2..=k).step_by(2) {
                                let e_a = ea[i * da + t] * ea[j * da + u] * ea[k * da + v];
                                for tt in (l % 2..=l).step_by(2) {
                                    for uu in (m % 2..=m).step_by(2) {
                                        for vv in (n % 2..=n).step_by(2) {
                                            sum += e_a
                                                * eb[l * db + tt]
                                                * eb[m * db + uu]
                                                * eb[n * db + vv]
                                                * table.get(t + tt, u + uu, v + vv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    block[ia * vb.ncart() + ib] += scale * sum;
                }
            }
        }
    }
    block
}

/// `Uaᵀ block Ub`: a Cartesian shell block in the spherical functions,
/// `nsph_a` rows by `nsph_b` columns at `ia * nsph_b + ib`.
fn to_spherical(block: &[f64], va: &ShellView, vb: &ShellView) -> Vec<f64> {
    let (nca, ncb, nsa, nsb) = (va.ncart(), vb.ncart(), va.nsph(), vb.nsph());
    // half[p][jb] = sum_q block[p][q] Ub[q][jb]
    let mut half = vec![0.0; nca * nsb];
    for p in 0..nca {
        for jb in 0..nsb {
            half[p * nsb + jb] = (0..ncb)
                .map(|q| block[p * ncb + q] * vb.harmonics[q + jb * ncb])
                .sum();
        }
    }
    let mut out = vec![0.0; nsa * nsb];
    for ia in 0..nsa {
        for jb in 0..nsb {
            out[ia * nsb + jb] = (0..nca)
                .map(|p| va.harmonics[p + ia * nca] * half[p * nsb + jb])
                .sum();
        }
    }
    out
}

/// `2 pi^(5/2)`, the two-center Coulomb integral's constant.
const TWO_PI_POW_5_2: f64 = 2.0 * 17.493_418_327_624_862;

/// `pi³`, the far field's: `2 pi^(5/2)` times `sqrt(pi) / 2`.
const PI_CUBED: f64 = PI * PI * PI;

/// Whether a pair of shells of total momentum `order`, `r2` apart
/// squared, is in the far field: `x = alpha r²` at least `40 + 4 order`
/// for its smallest `alpha`, so for every pair of its primitives.
///
/// What the far field drops is `exp(-x)` beside the Boys values. `F_n`
/// falls with `n` as `(2n-1)!! / (2x)^n`, so the term's relative size is
/// about `exp(-x) (2x)^(n-1) / (2n-1)!!`; at this bound that is under
/// 1e-17 for every order to 20 (4e-18 at 0, 6e-22 at 6, 1e-25 at 12), and
/// `erf(sqrt(x))` is 1 from 36.
fn is_far(alpha_min: f64, r2: f64, order: usize) -> bool {
    alpha_min * r2 >= 40.0 + 4.0 * order as f64
}

/// One shell as the direct route reads it, the pair-independent factors
/// computed once.
struct ShellData {
    /// The angular momentum.
    l: usize,
    /// The center, in bohr.
    center: [f64; 3],
    /// The shell's first spherical function.
    row0: usize,
    /// Each primitive's exponent `a` and `c / (a (2a)^l)`: its coefficient
    /// with Hobson's `1 / (2a)^l` and the integral's `1 / a` folded in.
    primitives: Vec<(f64, f64)>,
    /// The smallest exponent.
    min_exponent: f64,
    /// `sum c / (a^(3/2) (2a)^l)`: the shell's far-field strength, its
    /// multipole moment up to the harmonic's own factor.
    strength: f64,
}

impl ShellData {
    /// Shell `index` of `aux`.
    fn new(aux: &BasisSet, index: usize) -> ShellData {
        let shell = &aux.shells()[index];
        let l = shell.angular_momentum();
        let c = shell.center();
        let primitives: Vec<(f64, f64)> = shell
            .primitives()
            .iter()
            .map(|p| {
                let a = p.exponent;
                (a, p.coefficient / (a * (2.0 * a).powi(l)))
            })
            .collect();
        ShellData {
            l: l as usize,
            center: [c.x, c.y, c.z],
            row0: aux.ao_offset_sphe(index as u32) as usize,
            min_exponent: primitives.iter().fold(f64::INFINITY, |m, p| m.min(p.0)),
            strength: primitives.iter().map(|(a, w)| w / a.sqrt()).sum(),
            primitives,
        }
    }

    /// The spherical function count.
    fn nsph(&self) -> usize {
        2 * self.l + 1
    }
}

/// A run of consecutive shells on one center, an atom's: its shells, its
/// spherical functions and the largest momentum among them.
struct Center {
    /// The shells.
    shells: std::ops::Range<usize>,
    /// The spherical functions, the rows of `M` the center owns.
    rows: std::ops::Range<usize>,
    /// The largest momentum.
    l_max: usize,
}

/// The shells of `data` grouped into [`Center`]s, in order.
fn centers(data: &[ShellData], n: usize) -> Vec<Center> {
    let mut out: Vec<Center> = Vec::new();
    for (index, shell) in data.iter().enumerate() {
        match out.last_mut() {
            Some(last) if data[last.shells.start].center == shell.center => {
                last.shells.end = index + 1;
                last.l_max = last.l_max.max(shell.l);
            }
            _ => out.push(Center {
                shells: index..index + 1,
                rows: shell.row0..n,
                l_max: shell.l,
            }),
        }
    }
    for i in 1..out.len() {
        out[i - 1].rows.end = out[i].rows.start;
    }
    out
}

/// The packed triangle `m` cut into the rows each of `bounds` owns, the
/// bounds consecutive from row 0: disjoint `&mut` slices, one per task.
fn split_rows(
    m: &mut [f64],
    bounds: impl Iterator<Item = std::ops::Range<usize>>,
) -> Vec<&mut [f64]> {
    let mut rows = Vec::new();
    let mut rest = m;
    for bound in bounds {
        let (mine, later) =
            rest.split_at_mut(lower_index(bound.end, 0) - lower_index(bound.start, 0));
        rows.push(mine);
        rest = later;
    }
    rows
}

/// What a near pair's block is built in.
struct NearScratch {
    /// The Hermite Coulomb integrals of one primitive pair.
    table: HermiteCoulomb,
    /// The sum over primitive pairs of `weight * R_tuv`,
    /// `t + u + v = la + lb`, at `t * (la + lb + 1) + u`.
    top: Vec<f64>,
    /// The spherical block, `nsph_a` rows by `nsph_b` columns at
    /// `ia * nsph_b + ib`.
    out: Vec<f64>,
}

/// What one thread builds blocks in: allocated once per thread
/// ([`SCRATCH`]), not per matrix and never per pair. One built for a
/// momentum serves every lower one.
struct Scratch {
    /// A near pair's.
    near: NearScratch,
    /// The derivatives of `1 / |r|` of one pair of centers.
    far_table: HermiteCoulomb,
    /// The far-field blocks of one pair of centers, one per `(la, lb)` at
    /// `la * side + lb`, laid out as a near pair's.
    far_blocks: Vec<Vec<f64>>,
    /// Which of `far_blocks` are this pair of centers'.
    far_ready: Vec<bool>,
    /// `far_blocks`' side, the largest momentum plus one.
    side: usize,
}

thread_local! {
    /// Each thread's [`Scratch`]. The tables are a few hundred KiB at i, so
    /// building them per rayon job cost more than a small matrix did.
    static SCRATCH: std::cell::RefCell<Option<Scratch>> = const { std::cell::RefCell::new(None) };
}

impl Scratch {
    /// Scratch for shells up to momentum `l_max`.
    fn new(l_max: usize) -> Scratch {
        let nsph = 2 * l_max + 1;
        let d = 2 * l_max + 1;
        let side = l_max + 1;
        Scratch {
            near: NearScratch {
                table: HermiteCoulomb::new(2 * l_max),
                top: vec![0.0; d * d],
                out: vec![0.0; nsph * nsph],
            },
            far_table: HermiteCoulomb::new(2 * l_max),
            far_blocks: vec![vec![0.0; nsph * nsph]; side * side],
            far_ready: vec![false; side * side],
            side,
        }
    }
}

/// `sum over the two columns' nonzero terms of c_a c_b get(t, u)`, for
/// every pair of spherical functions of the two momenta, into `out`:
/// what turns the top-order Hermite integrals into the spherical block.
fn harmonic_sum(
    va: &ShellView,
    vb: &ShellView,
    scale: f64,
    get: impl Fn(usize, usize) -> f64,
    out: &mut [f64],
) {
    let nsb = vb.nsph();
    for (ia, row) in out[..va.nsph() * nsb].chunks_exact_mut(nsb).enumerate() {
        let column_a = va.column(ia);
        for (jb, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for &(ta, ua, c_a) in column_a {
                let mut inner = 0.0;
                for &(tb, ub, c_b) in vb.column(jb) {
                    inner = c_b.mul_add(get(ta + tb, ua + ub), inner);
                }
                sum += c_a * inner;
            }
            *value = scale * sum;
        }
    }
}

/// The spherical block `(a|b)` of two shells `r` apart with no Cartesian
/// integral formed, into `scratch.out`.
///
/// A column of a shell's Cartesian-to-spherical block is a harmonic
/// polynomial `H`, and for one of degree `l` (Hobson's theorem)
/// `H(r) exp(-a r²) = H(∇) exp(-a r²) / (2a)^l`: only the Hermite
/// Gaussians of the top order `t + u + v = l` enter, with `H`'s own
/// coefficients. So the Hermite expansion drops out, a primitive pair
/// contributes `R_tuv` of the one order `la + lb` under a scalar weight,
/// and an element of the block is a sum over the two columns' nonzero
/// terms of a lookup into that sum: no Cartesian-indexed matrix is formed
/// and no dense transform runs.
///
/// Two s shells are one Boys value per primitive pair, with no table.
fn near_block(
    a: &ShellData,
    va: &ShellView,
    b: &ShellData,
    vb: &ShellView,
    r: [f64; 3],
    scratch: &mut NearScratch,
) {
    let NearScratch { table, top, out } = scratch;
    let order = a.l + b.l;
    if order == 0 {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let mut sum = 0.0;
        let mut f = [0.0];
        for &(ax, wa) in &a.primitives {
            for &(bx, wb) in &b.primitives {
                boys(ax * bx / (ax + bx) * r2, &mut f);
                sum += wa * wb / (ax + bx).sqrt() * f[0];
            }
        }
        out[0] = TWO_PI_POW_5_2 * sum;
        return;
    }
    let d = order + 1;
    let top = &mut top[..d * d];
    top.fill(0.0);
    for &(ax, wa) in &a.primitives {
        for &(bx, wb) in &b.primitives {
            table.fill(order, ax * bx / (ax + bx), r);
            let weight = wa * wb / (ax + bx).sqrt();
            for t in 0..=order {
                for u in 0..=order - t {
                    top[t * d + u] = weight.mul_add(table.get(t, u, order - t - u), top[t * d + u]);
                }
            }
        }
    }
    let sign = if b.l.is_multiple_of(2) { 1.0 } else { -1.0 };
    harmonic_sum(va, vb, sign * TWO_PI_POW_5_2, |t, u| top[t * d + u], out);
}

/// The views of every momentum `aux` holds, by `l`.
fn views(aux: &BasisSet) -> Vec<ShellView> {
    let l_max = aux
        .shells()
        .iter()
        .map(Shell::angular_momentum)
        .max()
        .unwrap_or(0);
    (0..=l_max).map(ShellView::new).collect()
}

/// The lower triangle of the Coulomb metric `M[P][Q] = (P|Q)` of the
/// fitting set `aux` over its spherical layout, packed:
/// `n (n + 1) / 2` values for `n = nao_sphe`, `(P, Q)`, `Q <= P`, at
/// [`lower_index`]`(P, Q)`. `M` is symmetric, so the upper triangle is
/// never computed and never stored.
///
/// Each shell block is formed directly in the solid harmonics (the
/// module's "Directly spherical"), and a pair of shells far enough apart
/// from one block per pair of centers and momenta (its "The far field").
/// The coefficients are used as `aux` holds them; place the set
/// `primitive_normalized()`.
///
/// A fresh triangle: [`metric_into`] into a `Vec` of its own, for a caller
/// with no buffer to hand, a test or a reference.
#[must_use]
pub fn metric(aux: &BasisSet) -> Vec<f64> {
    let mut m = vec![0.0; lower_index(aux.nao_sphe() as usize, 0)];
    metric_into(aux, &mut m);
    m
}

/// [`metric`] written into the caller's `m`, every value of it, so a step
/// builds the triangle in its staging and allocates nothing of its size.
///
/// One rayon task per center. Its functions' rows are one contiguous
/// slice of the packed triangle, so the tasks write disjoint `&mut`
/// slices and share only what they read; each thread has its own scratch,
/// which it keeps between calls.
#[debug_requires(
    m.len() == lower_index(aux.nao_sphe() as usize, 0),
    "m is the n (n + 1) / 2 lower triangle"
)]
pub fn metric_into(aux: &BasisSet, m: &mut [f64]) {
    let n = aux.nao_sphe() as usize;
    let views = views(aux);
    let data: Vec<ShellData> = (0..aux.shells().len())
        .map(|index| ShellData::new(aux, index))
        .collect();
    let centers = centers(&data, n);
    let rows = split_rows(m, centers.iter().map(|c| c.rows.clone()));

    let l_max = views.len() - 1;
    rows.into_par_iter().enumerate().for_each(|(ca, rows)| {
        rows.fill(0.0);
        SCRATCH.with_borrow_mut(|slot| {
            // This thread's scratch, kept from task to task and call to
            // call; replaced only by one for a higher momentum.
            if slot.as_ref().is_none_or(|s| s.side <= l_max) {
                *slot = Some(Scratch::new(l_max));
            }
            let Some(scratch) = slot else { unreachable!() };
            let Scratch {
                near,
                far_table,
                far_blocks,
                far_ready,
                side,
            } = scratch;
            let center_a = &centers[ca];
            let start = lower_index(center_a.rows.start, 0);
            let at_a = data[center_a.shells.start].center;
            for (cb, center_b) in centers[..=ca].iter().enumerate() {
                let at_b = data[center_b.shells.start].center;
                let r = [at_a[0] - at_b[0], at_a[1] - at_b[1], at_a[2] - at_b[2]];
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                let mut far_filled = false;
                far_ready.fill(false);
                for sa in center_a.shells.clone() {
                    let a = &data[sa];
                    let va = &views[a.l];
                    // A center with itself: the shell pairs `b <= a`.
                    let last = if cb == ca {
                        sa + 1
                    } else {
                        center_b.shells.end
                    };
                    for (sb, b) in data[..last].iter().enumerate().skip(center_b.shells.start) {
                        let vb = &views[b.l];
                        let (nsa, nsb) = (a.nsph(), b.nsph());
                        let alpha_min =
                            a.min_exponent * b.min_exponent / (a.min_exponent + b.min_exponent);
                        if is_far(alpha_min, r2, a.l + b.l) {
                            if !far_filled {
                                far_table.fill_far(center_a.l_max + center_b.l_max, r);
                                far_filled = true;
                            }
                            let slot = a.l * *side + b.l;
                            if !far_ready[slot] {
                                let order = a.l + b.l;
                                let sign = if b.l.is_multiple_of(2) { 1.0 } else { -1.0 };
                                harmonic_sum(
                                    va,
                                    vb,
                                    sign * PI_CUBED,
                                    |t, u| far_table.get(t, u, order - t - u),
                                    &mut far_blocks[slot],
                                );
                                far_ready[slot] = true;
                            }
                            let scale = a.strength * b.strength;
                            let block = &far_blocks[slot];
                            for ia in 0..nsa {
                                let at = lower_index(a.row0 + ia, b.row0) - start;
                                for (to, from) in rows[at..at + nsb]
                                    .iter_mut()
                                    .zip(&block[ia * nsb..(ia + 1) * nsb])
                                {
                                    *to = scale * from;
                                }
                            }
                        } else {
                            near_block(a, va, b, vb, r, near);
                            for ia in 0..nsa {
                                // Shells are laid out in order, so an
                                // earlier shell's columns are all below
                                // the diagonal.
                                let cols = if sa == sb { ia + 1 } else { nsb };
                                let at = lower_index(a.row0 + ia, b.row0) - start;
                                rows[at..at + cols]
                                    .copy_from_slice(&near.out[ia * nsb..ia * nsb + cols]);
                            }
                        }
                    }
                }
            }
        });
    });
}

/// How many of `aux`'s shell pairs `A >= B` [`metric`] takes from the far
/// field, and how many pairs there are.
#[must_use]
pub fn far_pairs(aux: &BasisSet) -> (usize, usize) {
    let data: Vec<ShellData> = (0..aux.shells().len())
        .map(|index| ShellData::new(aux, index))
        .collect();
    let far: usize = data
        .par_iter()
        .enumerate()
        .map(|(sa, a)| {
            data[..sa]
                .iter()
                .filter(|b| {
                    let r2: f64 = (0..3).map(|k| (a.center[k] - b.center[k]).powi(2)).sum();
                    let alpha_min =
                        a.min_exponent * b.min_exponent / (a.min_exponent + b.min_exponent);
                    is_far(alpha_min, r2, a.l + b.l)
                })
                .count()
        })
        .sum();
    (far, data.len() * (data.len() + 1) / 2)
}

/// [`metric`] by the textbook route, kept to check and time it against:
/// every Cartesian integral `(a|b)` of a shell pair through the full
/// Hermite expansion, then `Uaᵀ (a|b) Ub`, with no far field. The same
/// packed triangle to rounding. One rayon task per shell.
#[must_use]
pub fn metric_via_cartesian(aux: &BasisSet) -> Vec<f64> {
    let n = aux.nao_sphe() as usize;
    let shells = aux.shells();
    let views = views(aux);
    let row0 = |shell: usize| {
        if shell == shells.len() {
            n
        } else {
            aux.ao_offset_sphe(shell as u32) as usize
        }
    };
    let mut m = vec![0.0; lower_index(n, 0)];
    let rows = split_rows(&mut m, (0..shells.len()).map(|s| row0(s)..row0(s + 1)));

    rows.into_par_iter().enumerate().for_each_init(
        || HermiteCoulomb::new(2 * (views.len() - 1)),
        |table, (sa, rows)| {
            let a = &shells[sa];
            let va = &views[a.angular_momentum() as usize];
            let start = lower_index(row0(sa), 0);
            for (sb, b) in shells[..=sa].iter().enumerate() {
                let vb = &views[b.angular_momentum() as usize];
                let block = to_spherical(&cartesian_block(a, va, b, vb, table), va, vb);
                let nsb = vb.nsph();
                for ia in 0..va.nsph() {
                    let cols = if sa == sb { ia + 1 } else { nsb };
                    let at = lower_index(row0(sa) + ia, row0(sb)) - start;
                    rows[at..at + cols].copy_from_slice(&block[ia * nsb..ia * nsb + cols]);
                }
            }
        },
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `F_n(x)` by composite Simpson over `t` in `[0, 1]`: an independent
    /// evaluation, good to about 1e-13 here.
    fn boys_by_quadrature(n: usize, x: f64) -> f64 {
        let steps = 20_000;
        let h = 1.0 / steps as f64;
        let g = |t: f64| t.powi(2 * n as i32) * (-x * t * t).exp();
        let mut sum = g(0.0) + g(1.0);
        for i in 1..steps {
            sum += g(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// Both branches against quadrature, for every highest order the
    /// tables cover: at zero, inside the series, on either side of that
    /// order's switch at `M + 8`, and far out.
    #[test]
    fn boys_matches_quadrature() {
        for top in 0..=BOYS_MAX_ORDER {
            let switch = (top + 8) as f64;
            let mut f = vec![0.0; top + 1];
            for x in [
                0.0,
                1e-9,
                0.3,
                2.0,
                0.5 * switch,
                switch - 1e-9,
                switch,
                60.0,
                400.0,
            ] {
                boys(x, &mut f);
                for (n, &got) in f.iter().enumerate() {
                    let want = boys_by_quadrature(n, x);
                    assert!(
                        (got - want).abs() <= 1e-11 * want,
                        "top {top}: F_{n}({x}) = {got:e}, quadrature {want:e}"
                    );
                }
            }
        }
    }

    /// `F_n(0) = 1 / (2n + 1)`.
    #[test]
    fn boys_at_zero_is_exact() {
        let mut f = [0.0; 13];
        boys(0.0, &mut f);
        for (n, &got) in f.iter().enumerate() {
            assert!((got - 1.0 / (2 * n + 1) as f64).abs() <= 1e-16);
        }
    }

    /// `x² exp(-a x²) = Λ_0 / (2a) + Λ_2 / (4a²)`, and `x³` has no even
    /// term.
    #[test]
    fn hermite_expansion_of_low_powers() {
        let a = 1.7;
        let e = hermite_expansion(3, a);
        assert_eq!(e[0], 1.0);
        assert_eq!(e[4 + 1], 0.5 / a);
        assert!((e[2 * 4] - 0.5 / a).abs() <= 1e-16);
        assert!((e[2 * 4 + 2] - 0.25 / (a * a)).abs() <= 1e-16);
        assert_eq!(e[3 * 4], 0.0);
        assert_eq!(e[3 * 4 + 2], 0.0);
    }

    /// Two normalized s primitives: `(a|b) = N_a N_b 2 pi^(5/2) /
    /// (a b sqrt(a + b)) F_0(alpha R²)`, on one center and apart.
    #[test]
    fn two_s_primitives_have_the_closed_form() {
        use fb_basis::ShellStamp;

        let (a, b) = (0.9, 2.3);
        let place = |exponent: f64, z: f64| {
            let mut center = Shell::zipped(0, &[1.0], &[1.0], Default::default()).center();
            center.z = z;
            ShellStamp::zipped(0, &[exponent], &[1.0])
                .primitive_normalized()
                .place(center)
        };
        let norm = |x: f64| (2.0 * x / PI).powf(0.75);
        for z in [0.0, 1.4] {
            let aux = BasisSet::new(vec![place(a, 0.0), place(b, z)]);
            let m = metric(&aux);
            let via = metric_via_cartesian(&aux);
            assert!((m[1] - via[1]).abs() <= 1e-15 * via[1]);
            let mut f = [0.0];
            boys(a * b / (a + b) * z * z, &mut f);
            let want = norm(a) * norm(b) * 2.0 * PI.powf(2.5) / (a * b * (a + b).sqrt()) * f[0];
            assert!((TWO_PI_POW_5_2 - 2.0 * PI.powf(2.5)).abs() <= 1e-14);
            assert!((m[1] - want).abs() <= 1e-15 * want, "{} vs {want}", m[1]);
            assert_eq!(m.len(), 3);
        }
    }

    /// The far field against the full integral just past its bound, where
    /// what it drops is largest: two contracted shells of every pair of
    /// momenta through i, off every axis. The reference is `near_block`,
    /// the same pair with the Boys function and no far field; the bar is
    /// 1e-13 of the block's largest element.
    #[test]
    fn far_field_matches_the_full_integral_at_its_bound() {
        use fb_basis::ShellStamp;

        let (ea, eb) = ([0.8, 2.1], [1.3, 0.5]);
        let alpha_min = 0.8 * 0.5 / 1.3;
        for la in 0..=6 {
            for lb in 0..=6 {
                let order = (la + lb) as usize;
                let distance = ((40.0 + 4.0 * order as f64) / alpha_min).sqrt() * 1.0001;
                let direction = [0.48, -0.6, 0.64];
                let mut center = Shell::zipped(0, &[1.0], &[1.0], Default::default()).center();
                let origin = center;
                center.x = distance * direction[0];
                center.y = distance * direction[1];
                center.z = distance * direction[2];
                let aux = BasisSet::new(vec![
                    ShellStamp::zipped(lb, &eb, &[0.7, 0.4])
                        .primitive_normalized()
                        .place(origin),
                    ShellStamp::zipped(la, &ea, &[0.3, 0.9])
                        .primitive_normalized()
                        .place(center),
                ]);
                assert_eq!(far_pairs(&aux), (1, 3), "({la}|{lb}) is a far pair");
                let m = metric(&aux);

                let views = views(&aux);
                let (b, a) = (ShellData::new(&aux, 0), ShellData::new(&aux, 1));
                let mut scratch = Scratch::new(views.len() - 1);
                let r = [center.x, center.y, center.z];
                near_block(&a, &views[a.l], &b, &views[b.l], r, &mut scratch.near);
                let (nsa, nsb) = (a.nsph(), b.nsph());
                let want = &scratch.near.out[..nsa * nsb];
                let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(scale > 0.0);
                for ia in 0..nsa {
                    for jb in 0..nsb {
                        let got = m[lower_index(a.row0 + ia, b.row0 + jb)];
                        let error = (got - want[ia * nsb + jb]).abs();
                        assert!(
                            error <= 1e-13 * scale,
                            "({la}|{lb}) [{ia}][{jb}]: {got:e} vs {:e}, block max {scale:e}",
                            want[ia * nsb + jb]
                        );
                    }
                }
            }
        }
    }
}
