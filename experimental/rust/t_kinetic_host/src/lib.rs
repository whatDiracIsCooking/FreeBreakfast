//! The spherical AO kinetic energy `T[μ][ν] = <μ| -∇²/2 |ν>` of a basis,
//! computed on the host: plain Rust on rayon's threads, no device and no
//! kernel library.
//!
//! **The integral.** Two primitive s Gaussians of exponents `a` and `b` on
//! centers `A` and `B`, `R = A - B` apart, overlap as
//! `S(R) = (pi / p)^(3/2) exp(-mu R²)`, `p = a + b`, `mu = a b / p`. The
//! Laplacian of the ket is the Laplacian in `B`, which is the Laplacian in
//! `R`, and it passes outside the integral:
//!
//! ```text
//! T(R) = -∇²_R S(R) / 2 = mu (3 - 2 mu R²) (pi / p)^(3/2) exp(-mu R²)
//! ```
//!
//! **The layout.** `T` is over the `2l+1` real solid harmonics of each
//! shell, through the basis's own Cartesian-to-spherical block
//! ([`fb_basis::solid_harmonic_block`], scaled by `sqrt((2l-1)!!)` as
//! [`fb_basis::spherical_transform_matrix`] scales it). Coefficients are
//! taken as given: place the basis `primitive_normalized()`, as the
//! pipeline does, and the result is the tree's T, contractions never
//! renormalized.
//!
//! **The lower triangle alone.** `T` is symmetric, and [`kinetic`] returns
//! it packed: `(μ, ν)`, `ν <= μ`, at [`lower_index`], `n (n + 1) / 2`
//! values. Only the shell pairs `A >= B` are computed, one rayon task per
//! center, whose rows are one contiguous slice of the triangle
//! (`RAYON_NUM_THREADS` sets the thread count).
//!
//! **Directly spherical.** [`kinetic`] never forms a Cartesian integral. A
//! column of a shell's harmonic block is a harmonic polynomial `H` of
//! degree `l`, and Hobson's theorem gives
//! `H(r - A) exp(-a |r - A|²) = H(∇_A) exp(-a |r - A|²) / (2a)^l`. Both
//! derivatives pass outside the integral, `∇_B = -∇_R`, and they commute
//! with the Laplacian, so
//!
//! ```text
//! <a|T|b> = (-1)^lb / ((2a)^la (2b)^lb) * H_a(∇_R) H_b(∇_R) (-∇²_R / 2) S(R)
//! ```
//!
//! A Gaussian's derivatives factor over the axes:
//! `∂^t_X exp(-mu X²) = h_t(X) exp(-mu X²)`, `h_0 = 1`,
//! `h_(t+1) = -2 mu (X h_t + t h_(t-1))`. So a contracted shell pair sums,
//! for each `t + u + v = la + lb`, `h_(t+2) h_u h_v + h_t h_(u+2) h_v +
//! h_t h_u h_(v+2)` under a scalar weight over its primitive pairs: the
//! overlap's sum with two more orders of the same three 1-D recurrences.
//! An element of the block is a sum over the two harmonics' nonzero
//! Cartesian terms of a lookup into that sum. [`kinetic_via_cartesian`] is
//! the textbook route, Obara-Saika per axis and a dense transform, kept to
//! check and time against.
//!
//! **S beside it.** The overlap is the order-`la + lb` sum of the same
//! `h_t` under the same weight, so [`overlap_and_kinetic`] returns both
//! triangles from one exponential, weight and set of recurrences per
//! primitive pair.
//!
//! **Negligible pairs.** A primitive pair's block is `mu` times a
//! polynomial of degree `la + lb + 2` in `sqrt(mu) R` times
//! `exp(-mu R²)`, so a primitive pair with
//! `mu |A - B|² >= 40 + 4 (la + lb + 2)` is skipped, and a pair of shells
//! past it for its smallest `mu` is left zero (see [`is_negligible`]). The
//! first test is the one that pays: between atoms, the tight primitives of
//! a shell pair that is kept are mostly past it.

#![forbid(unsafe_code)]

// The hot loops are `f64::mul_add`, which is one instruction with the FMA
// target feature and a software call without it. `.cargo/config.toml` turns
// the feature on; this refuses a build that lost it (cargo run from another
// directory, or a RUSTFLAGS in the environment, which replaces the file's).
#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
compile_error!(
    "fb-t-kinetic-host needs the FMA target feature: build from its own directory, \
     whose .cargo/config.toml sets it, or add `-C target-feature=+fma` to RUSTFLAGS"
);

use std::f64::consts::PI;

use fb_basis::{BasisSet, Shell, cartesian_powers, solid_harmonic_block};
use fb_core_primitives::double_factorial_odd;
use rayon::prelude::*;

/// One shell's momentum as the integral loops read it: its Cartesian
/// powers, and its Cartesian-to-spherical block with the shell's scale
/// folded in, dense and by its nonzeros.
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

/// The views of every momentum `basis` holds, by `l`.
fn views(basis: &BasisSet) -> Vec<ShellView> {
    let l_max = basis
        .shells()
        .iter()
        .map(Shell::angular_momentum)
        .max()
        .unwrap_or(0);
    (0..=l_max).map(ShellView::new).collect()
}

/// Where `(μ, ν)`, `ν <= μ`, sits in the packed lower triangle: rows one
/// after another, row `μ` holding its `μ + 1` entries.
///
/// ```
/// use fb_t_kinetic_host::lower_index;
///
/// assert_eq!([lower_index(0, 0), lower_index(1, 0), lower_index(1, 1)], [0, 1, 2]);
/// assert_eq!(lower_index(2, 0), 3);
/// ```
#[must_use]
pub const fn lower_index(p: usize, q: usize) -> usize {
    p * (p + 1) / 2 + q
}

/// The packed triangle `t` cut into the rows each of `bounds` owns, the
/// bounds consecutive from row 0: disjoint `&mut` slices, one per task.
fn split_rows(
    t: &mut [f64],
    bounds: impl Iterator<Item = std::ops::Range<usize>>,
) -> Vec<&mut [f64]> {
    let mut rows = Vec::new();
    let mut rest = t;
    for bound in bounds {
        let (mine, later) =
            rest.split_at_mut(lower_index(bound.end, 0) - lower_index(bound.start, 0));
        rows.push(mine);
        rest = later;
    }
    rows
}

/// Whether a primitive pair of total momentum `order`, `r2` apart squared,
/// is negligible: `mu r²` at least `40 + 4 (order + 2)`. Given a pair of
/// shells' smallest `mu`, whether every pair of its primitives is.
///
/// A primitive pair of the block is `mu` times a polynomial of degree
/// `order + 2` in `sqrt(mu) R` times `exp(-mu R²)`, with
/// primitive-normalized factors below one. Past this bound that falls with
/// `mu`, so the smallest `mu` is the largest term
/// (`negligible_pairs_are_below_1e_15` checks through i, diffuse and
/// tight; `skipped_primitive_pairs_change_nothing` checks a kept pair
/// whose tight primitives are skipped).
fn is_negligible(mu: f64, r2: f64, order: usize) -> bool {
    mu * r2 >= 40.0 + 4.0 * (order + 2) as f64
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
    /// Each primitive's exponent `a` and `c / (2a)^l`: its coefficient
    /// with Hobson's `1 / (2a)^l` folded in.
    primitives: Vec<(f64, f64)>,
    /// The smallest exponent.
    min_exponent: f64,
}

impl ShellData {
    /// Shell `index` of `basis`.
    fn new(basis: &BasisSet, index: usize) -> ShellData {
        let shell = &basis.shells()[index];
        let l = shell.angular_momentum();
        let c = shell.center();
        let primitives: Vec<(f64, f64)> = shell
            .primitives()
            .iter()
            .map(|p| (p.exponent, p.coefficient / (2.0 * p.exponent).powi(l)))
            .collect();
        ShellData {
            l: l as usize,
            center: [c.x, c.y, c.z],
            row0: basis.ao_offset_sphe(index as u32) as usize,
            min_exponent: primitives.iter().fold(f64::INFINITY, |m, p| m.min(p.0)),
            primitives,
        }
    }

    /// The spherical function count.
    fn nsph(&self) -> usize {
        2 * self.l + 1
    }
}

/// The smallest `mu` of a pair of shells.
fn mu_min(a: &ShellData, b: &ShellData) -> f64 {
    a.min_exponent * b.min_exponent / (a.min_exponent + b.min_exponent)
}

/// A run of consecutive shells on one center, an atom's: its shells and
/// its spherical functions.
struct Center {
    /// The shells.
    shells: std::ops::Range<usize>,
    /// The spherical functions, the rows of `T` the center owns.
    rows: std::ops::Range<usize>,
}

/// The shells of `data` grouped into [`Center`]s, in order.
fn centers(data: &[ShellData], n: usize) -> Vec<Center> {
    let mut out: Vec<Center> = Vec::new();
    for (index, shell) in data.iter().enumerate() {
        match out.last_mut() {
            Some(last) if data[last.shells.start].center == shell.center => {
                last.shells.end = index + 1;
            }
            _ => out.push(Center {
                shells: index..index + 1,
                rows: shell.row0..n,
            }),
        }
    }
    for i in 1..out.len() {
        out[i - 1].rows.end = out[i].rows.start;
    }
    out
}

/// What a block is built in: a few hundred doubles at i, one per task.
struct Scratch {
    /// `h_t` of the three axes, `t <= la + lb + 2`, at `axis * (d + 2) + t`.
    h: Vec<f64>,
    /// The sum over primitive pairs of `weight (h_(t+2) h_u h_v + h_t
    /// h_(u+2) h_v + h_t h_u h_(v+2))`, `t + u + v = la + lb`, at
    /// `t * (la + lb + 1) + u`.
    top: Vec<f64>,
    /// The overlap's sum, `weight h_t h_u h_v`, laid out as `top`; filled
    /// only when the overlap is asked for.
    top_s: Vec<f64>,
    /// The spherical block of T, `nsph_a` rows by `nsph_b` columns at
    /// `ia * nsph_b + ib`.
    out: Vec<f64>,
    /// The spherical block of S, laid out as `out`; filled only when the
    /// overlap is asked for.
    out_s: Vec<f64>,
}

impl Scratch {
    /// Scratch for shells up to momentum `l_max`.
    fn new(l_max: usize) -> Scratch {
        let d = 2 * l_max + 1;
        Scratch {
            h: vec![0.0; 3 * (d + 2)],
            top: vec![0.0; d * d],
            top_s: vec![0.0; d * d],
            out: vec![0.0; d * d],
            out_s: vec![0.0; d * d],
        }
    }
}

/// `top`, the sum of a shell pair of total momentum `d - 1`, contracted
/// with the two harmonics into the spherical block `out`, `nsph_a` rows by
/// `nsph_b` columns at `ia * nsph_b + ib`.
fn contract(top: &[f64], d: usize, va: &ShellView, vb: &ShellView, out: &mut [f64]) {
    let sign = if vb.l.is_multiple_of(2) { 1.0 } else { -1.0 };
    let nsb = vb.nsph();
    for (ia, row) in out[..va.nsph() * nsb].chunks_exact_mut(nsb).enumerate() {
        let column_a = va.column(ia);
        for (jb, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for &(ta, ua, c_a) in column_a {
                let mut inner = 0.0;
                for &(tb, ub, c_b) in vb.column(jb) {
                    inner = c_b.mul_add(top[(ta + tb) * d + ua + ub], inner);
                }
                sum += c_a * inner;
            }
            *value = sign * sum;
        }
    }
}

/// The spherical block `<a|T|b>` of two shells `r = A - B` apart with no
/// Cartesian integral formed, into `scratch.out`, and with `OVERLAP` also
/// `<a|b>` into `scratch.out_s`, from the same exponentials, weights and
/// recurrences: the module's "Directly spherical". Negligible primitive
/// pairs, by T's bound, are skipped; the shell pair is not screened here.
fn block<const OVERLAP: bool>(
    a: &ShellData,
    va: &ShellView,
    b: &ShellData,
    vb: &ShellView,
    r: [f64; 3],
    scratch: &mut Scratch,
) {
    let Scratch {
        h,
        top,
        top_s,
        out,
        out_s,
    } = scratch;
    let order = a.l + b.l;
    let d = order + 1;
    let dh = d + 2;
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if order == 0 {
        let (mut t, mut s) = (0.0, 0.0);
        for &(ax, wa) in &a.primitives {
            for &(bx, wb) in &b.primitives {
                let p = ax + bx;
                let mu = ax * bx / p;
                if is_negligible(mu, r2, 0) {
                    continue;
                }
                let overlap = wa * wb * (PI / p) * (PI / p).sqrt() * (-mu * r2).exp();
                t = (overlap * mu).mul_add((2.0 * mu).mul_add(-r2, 3.0), t);
                if OVERLAP {
                    s += overlap;
                }
            }
        }
        out[0] = t;
        out_s[0] = s;
        return;
    }
    let top = &mut top[..d * d];
    top.fill(0.0);
    let top_s = &mut top_s[..d * d];
    if OVERLAP {
        top_s.fill(0.0);
    }
    for &(ax, wa) in &a.primitives {
        for &(bx, wb) in &b.primitives {
            let p = ax + bx;
            let mu = ax * bx / p;
            if is_negligible(mu, r2, order) {
                continue;
            }
            for (axis, &x) in r.iter().enumerate() {
                let h = &mut h[axis * dh..(axis + 1) * dh];
                h[0] = 1.0;
                h[1] = -2.0 * mu * x;
                for t in 1..order + 2 {
                    h[t + 1] = -2.0 * mu * (t as f64).mul_add(h[t - 1], x * h[t]);
                }
            }
            let weight = wa * wb * (PI / p) * (PI / p).sqrt() * (-mu * r2).exp();
            // `-∇²/2`'s half folded into T's weight.
            let weight_t = -0.5 * weight;
            let (hx, rest) = h.split_at(dh);
            let (hy, hz) = rest.split_at(dh);
            for t in 0..=order {
                for u in 0..=order - t {
                    let v = order - t - u;
                    let laplacian = hx[t + 2].mul_add(
                        hy[u] * hz[v],
                        hx[t] * hy[u + 2].mul_add(hz[v], hy[u] * hz[v + 2]),
                    );
                    top[t * d + u] = weight_t.mul_add(laplacian, top[t * d + u]);
                    if OVERLAP {
                        top_s[t * d + u] =
                            (weight * hx[t]).mul_add(hy[u] * hz[v], top_s[t * d + u]);
                    }
                }
            }
        }
    }
    contract(top, d, va, vb, out);
    if OVERLAP {
        contract(top_s, d, va, vb, out_s);
    }
}

/// The lower triangle of the kinetic energy `T[μ][ν] = <μ| -∇²/2 |ν>` of
/// `basis` over its spherical layout, packed: `n (n + 1) / 2` values for
/// `n = nao_sphe`, `(μ, ν)`, `ν <= μ`, at [`lower_index`]`(μ, ν)`. `T` is
/// symmetric, so the upper triangle is never computed and never stored.
///
/// Each shell block is formed directly in the solid harmonics (the
/// module's "Directly spherical"); a negligible pair is left zero. The
/// coefficients are used as `basis` holds them; place it
/// `primitive_normalized()`.
///
/// One rayon task per center. Its functions' rows are one contiguous slice
/// of the packed triangle, so the tasks write disjoint `&mut` slices and
/// share only what they read.
#[must_use]
pub fn kinetic(basis: &BasisSet) -> Vec<f64> {
    triangles::<false>(basis).1
}

/// The lower triangles of the overlap `S` and of the kinetic energy `T`,
/// in that order, both packed as [`kinetic`] packs `T`, in one pass:
/// each primitive pair's exponential, weight and recurrences serve both,
/// so the pair costs little more than `T` alone.
///
/// `T` carries the same bits as [`kinetic`]'s. `S` is screened by `T`'s
/// bound, which is past `S`'s own, so it drops nothing an overlap-only
/// route would keep.
#[must_use]
pub fn overlap_and_kinetic(basis: &BasisSet) -> (Vec<f64>, Vec<f64>) {
    triangles::<true>(basis)
}

/// [`overlap_and_kinetic`], with `S` left empty unless `OVERLAP`.
fn triangles<const OVERLAP: bool>(basis: &BasisSet) -> (Vec<f64>, Vec<f64>) {
    let n = basis.nao_sphe() as usize;
    let views = views(basis);
    let data: Vec<ShellData> = (0..basis.shells().len())
        .map(|index| ShellData::new(basis, index))
        .collect();
    let centers = centers(&data, n);
    let bounds = || centers.iter().map(|c| c.rows.clone());
    // `vec![0.0; ..]`, not a clone of `t`: zeroed pages from the allocator,
    // touched first by the task that owns them, where a clone writes the
    // whole triangle on one thread before any task starts.
    let mut t = vec![0.0; lower_index(n, 0)];
    let mut s = if OVERLAP {
        vec![0.0; lower_index(n, 0)]
    } else {
        Vec::new()
    };
    let rows_t = split_rows(&mut t, bounds());
    let rows_s = if OVERLAP {
        split_rows(&mut s, bounds())
    } else {
        bounds().map(|_| <&mut [f64]>::default()).collect()
    };

    let l_max = views.len() - 1;
    rows_t
        .into_par_iter()
        .zip(rows_s)
        .enumerate()
        .for_each(|(ca, (rows_t, rows_s))| {
            let mut scratch = Scratch::new(l_max);
            let center_a = &centers[ca];
            let start = lower_index(center_a.rows.start, 0);
            let at_a = data[center_a.shells.start].center;
            for (cb, center_b) in centers[..=ca].iter().enumerate() {
                let at_b = data[center_b.shells.start].center;
                let r = [at_a[0] - at_b[0], at_a[1] - at_b[1], at_a[2] - at_b[2]];
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                for sa in center_a.shells.clone() {
                    let a = &data[sa];
                    // A center with itself: the shell pairs `b <= a`.
                    let last = if cb == ca {
                        sa + 1
                    } else {
                        center_b.shells.end
                    };
                    for (sb, b) in data[..last].iter().enumerate().skip(center_b.shells.start) {
                        if is_negligible(mu_min(a, b), r2, a.l + b.l) {
                            continue;
                        }
                        block::<OVERLAP>(a, &views[a.l], b, &views[b.l], r, &mut scratch);
                        let nsb = b.nsph();
                        for ia in 0..a.nsph() {
                            // Shells are laid out in order, so an earlier
                            // shell's columns are all below the diagonal.
                            let cols = if sa == sb { ia + 1 } else { nsb };
                            let at = lower_index(a.row0 + ia, b.row0) - start;
                            let from = ia * nsb..ia * nsb + cols;
                            rows_t[at..at + cols].copy_from_slice(&scratch.out[from.clone()]);
                            if OVERLAP {
                                rows_s[at..at + cols].copy_from_slice(&scratch.out_s[from]);
                            }
                        }
                    }
                }
            }
        });
    (s, t)
}

/// How many of `basis`'s shell pairs `A >= B` [`kinetic`] leaves zero as
/// negligible, and how many pairs there are.
#[must_use]
pub fn negligible_pairs(basis: &BasisSet) -> (usize, usize) {
    let data: Vec<ShellData> = (0..basis.shells().len())
        .map(|index| ShellData::new(basis, index))
        .collect();
    let negligible: usize = data
        .par_iter()
        .enumerate()
        .map(|(sa, a)| {
            data[..sa]
                .iter()
                .filter(|b| {
                    let r2: f64 = (0..3).map(|k| (a.center[k] - b.center[k]).powi(2)).sum();
                    is_negligible(mu_min(a, b), r2, a.l + b.l)
                })
                .count()
        })
        .sum();
    (negligible, data.len() * (data.len() + 1) / 2)
}

/// The one-axis Obara-Saika overlaps `s_ij`, `i <= la`, `j <= lb`, of
/// `x_A^i x_B^j exp(-a x_A² - b x_B²)`, at `i * (lb + 1) + j`:
/// `s_00 = sqrt(pi / p) exp(-mu X²)`,
/// `s_(i+1)j = X_PA s_ij + (i s_(i-1)j + j s_i(j-1)) / (2p)` and its `j`
/// twin.
fn obara_saika(la: usize, lb: usize, a: f64, b: f64, xa: f64, xb: f64) -> Vec<f64> {
    let p = a + b;
    let xp = (a * xa + b * xb) / p;
    let (pa, pb) = (xp - xa, xp - xb);
    let half = 0.5 / p;
    let d = lb + 1;
    let mut s = vec![0.0; (la + 1) * d];
    s[0] = (PI / p).sqrt() * (-a * b / p * (xa - xb).powi(2)).exp();
    for j in 0..lb {
        let below = if j > 0 { s[j - 1] } else { 0.0 };
        s[j + 1] = pb * s[j] + half * j as f64 * below;
    }
    for i in 0..la {
        for j in 0..=lb {
            let down_i = if i > 0 { s[(i - 1) * d + j] } else { 0.0 };
            let down_j = if j > 0 { s[i * d + j - 1] } else { 0.0 };
            s[(i + 1) * d + j] = pa * s[i * d + j] + half * (i as f64 * down_i + j as f64 * down_j);
        }
    }
    s
}

/// The one-axis kinetic integrals `t_ij = <i| -d²/dx² / 2 |j>`, `i <= la`,
/// `j <= lb`, at `i * (lb + 1) + j`, from the overlaps `s` of
/// [`obara_saika`]`(la, lb + 2, ..)`: the ket differentiated twice,
/// `t_ij = b (2j + 1) s_ij - 2 b² s_i(j+2) - j (j - 1) s_i(j-2) / 2`.
fn kinetic_axis(la: usize, lb: usize, b: f64, s: &[f64]) -> Vec<f64> {
    let (d, ds) = (lb + 1, lb + 3);
    let mut t = vec![0.0; (la + 1) * d];
    for i in 0..=la {
        for j in 0..=lb {
            let down = if j >= 2 { s[i * ds + j - 2] } else { 0.0 };
            t[i * d + j] = b * (2 * j + 1) as f64 * s[i * ds + j]
                - 2.0 * b * b * s[i * ds + j + 2]
                - 0.5 * (j * j.saturating_sub(1)) as f64 * down;
        }
    }
    t
}

/// The contracted Cartesian block `<a|T|b>` of two shells, `ncart_a` rows
/// by `ncart_b` columns at `ia * ncart_b + ib`, the coefficients as given:
/// `t_x s_y s_z + s_x t_y s_z + s_x s_y t_z`.
fn cartesian_block(a: &Shell, va: &ShellView, b: &Shell, vb: &ShellView) -> Vec<f64> {
    let (ca, cb) = (a.center(), b.center());
    let (ca, cb) = ([ca.x, ca.y, ca.z], [cb.x, cb.y, cb.z]);
    let ds = vb.l + 3;
    let d = vb.l + 1;
    let mut block = vec![0.0; va.ncart() * vb.ncart()];
    for pa in a.primitives() {
        for pb in b.primitives() {
            let s: Vec<Vec<f64>> = (0..3)
                .map(|k| obara_saika(va.l, vb.l + 2, pa.exponent, pb.exponent, ca[k], cb[k]))
                .collect();
            let t: Vec<Vec<f64>> = s
                .iter()
                .map(|s| kinetic_axis(va.l, vb.l, pb.exponent, s))
                .collect();
            let c = pa.coefficient * pb.coefficient;
            for (ia, &[i, j, k]) in va.powers.iter().enumerate() {
                for (ib, &[l, m, n]) in vb.powers.iter().enumerate() {
                    let (sx, sy, sz) = (s[0][i * ds + l], s[1][j * ds + m], s[2][k * ds + n]);
                    let (tx, ty, tz) = (t[0][i * d + l], t[1][j * d + m], t[2][k * d + n]);
                    block[ia * vb.ncart() + ib] += c * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
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

/// [`kinetic`] by the textbook route, kept to check and time it against:
/// every Cartesian `<a|T|b>` of a shell pair by Obara-Saika per axis, then
/// `Uaᵀ <a|T|b> Ub`, with no pair screened. The same packed triangle to
/// rounding. One rayon task per shell.
#[must_use]
pub fn kinetic_via_cartesian(basis: &BasisSet) -> Vec<f64> {
    let n = basis.nao_sphe() as usize;
    let shells = basis.shells();
    let views = views(basis);
    let row0 = |shell: usize| {
        if shell == shells.len() {
            n
        } else {
            basis.ao_offset_sphe(shell as u32) as usize
        }
    };
    let mut t = vec![0.0; lower_index(n, 0)];
    let rows = split_rows(&mut t, (0..shells.len()).map(|i| row0(i)..row0(i + 1)));

    rows.into_par_iter().enumerate().for_each(|(sa, rows)| {
        let a = &shells[sa];
        let va = &views[a.angular_momentum() as usize];
        let start = lower_index(row0(sa), 0);
        for (sb, b) in shells[..=sa].iter().enumerate() {
            let vb = &views[b.angular_momentum() as usize];
            let block = to_spherical(&cartesian_block(a, va, b, vb), va, vb);
            let nsb = vb.nsph();
            for ia in 0..va.nsph() {
                let cols = if sa == sb { ia + 1 } else { nsb };
                let at = lower_index(row0(sa) + ia, row0(sb)) - start;
                rows[at..at + cols].copy_from_slice(&block[ia * nsb..ia * nsb + cols]);
            }
        }
    });
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use fb_basis::ShellStamp;

    /// A primitive-normalized shell placed at `at`.
    fn place(l: i32, exponents: &[f64], coefficients: &[f64], at: [f64; 3]) -> Shell {
        let mut center = Shell::zipped(0, &[1.0], &[1.0], Default::default()).center();
        center.x = at[0];
        center.y = at[1];
        center.z = at[2];
        ShellStamp::zipped(l, exponents, coefficients)
            .primitive_normalized()
            .place(center)
    }

    /// Two normalized s primitives: `<a|T|b> = mu (3 - 2 mu R²) (4 a b /
    /// p²)^(3/4) exp(-mu R²)`, on one center and apart (past the node of
    /// `3 - 2 mu R²` too), by both routes.
    #[test]
    fn two_s_primitives_have_the_closed_form() {
        let (a, b) = (0.9, 2.3);
        let p = a + b;
        let mu = a * b / p;
        for z in [0.0, 0.7, 1.4] {
            let basis = BasisSet::new(vec![
                place(0, &[a], &[1.0], [0.0; 3]),
                place(0, &[b], &[1.0], [0.0, 0.0, z]),
            ]);
            let t = kinetic(&basis);
            let via = kinetic_via_cartesian(&basis);
            let want = mu
                * (3.0 - 2.0 * mu * z * z)
                * (4.0 * a * b / (p * p)).powf(0.75)
                * (-mu * z * z).exp();
            assert_eq!(t.len(), 3);
            assert!((t[0] - 1.5 * a).abs() <= 1e-15 && (t[2] - 1.5 * b).abs() <= 1e-14);
            for (route, got) in [("spherical", t[1]), ("via cartesian", via[1])] {
                assert!(
                    (got - want).abs() <= 1e-15,
                    "{route}, z {z}: {got} vs {want}"
                );
            }
        }
    }

    /// A normalized primitive `r^l Y_lm exp(-a r²)` of every momentum
    /// through i, alone on its center, has `<T> = a (l + 3/2)` in every
    /// spherical component and no coupling between components, by both
    /// routes.
    #[test]
    fn a_primitive_of_every_momentum_has_the_closed_form() {
        let a = 1.7;
        for l in 0..=6 {
            let basis = BasisSet::new(vec![place(l, &[a], &[1.0], [0.3, -0.2, 0.5])]);
            let want = a * (f64::from(l) + 1.5);
            for (route, t) in [
                ("spherical", kinetic(&basis)),
                ("via cartesian", kinetic_via_cartesian(&basis)),
            ] {
                for m in 0..(2 * l + 1) as usize {
                    for k in 0..m {
                        assert!(t[lower_index(m, k)].abs() <= 1e-13, "{route}, l {l}");
                    }
                    let diag = t[lower_index(m, m)];
                    assert!(
                        (diag - want).abs() <= 1e-13 * want,
                        "{route}, l {l}, m {m}: {diag} vs {want}"
                    );
                }
            }
        }
    }

    /// The two routes agree on contracted shells of every pair of momenta
    /// through i, off every axis, at a distance where the integral is
    /// large, to 1e-14 of the block's largest element.
    #[test]
    fn the_routes_agree_through_i() {
        for la in 0..=6 {
            for lb in 0..=6 {
                let basis = BasisSet::new(vec![
                    place(lb, &[1.3, 0.5], &[0.7, 0.4], [0.0; 3]),
                    place(la, &[0.8, 2.1], &[0.3, 0.9], [0.48, -0.6, 0.64]),
                ]);
                let (t, via) = (kinetic(&basis), kinetic_via_cartesian(&basis));
                let scale = via.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                for (k, (got, want)) in t.iter().zip(&via).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-14 * scale,
                        "<{la}|T|{lb}> [{k}]: {got:e} vs {want:e}"
                    );
                }
            }
        }
    }

    /// [`overlap_and_kinetic`] on contracted shells of every pair of
    /// momenta through i, off every axis, near (the overlap large) and at
    /// the distance where the tight primitive pairs are skipped: its `T`
    /// carries [`kinetic`]'s bits, and its `S` is `fb_s_overlap_host`'s to
    /// 1e-14.
    #[test]
    fn overlap_and_kinetic_match_each_alone_through_i() {
        for r in [[0.48, -0.6, 0.64], [0.96, -1.2, 1.28]] {
            for la in 0..=6 {
                for lb in 0..=6 {
                    let basis = BasisSet::new(vec![
                        place(lb, &[900.0, 1.3, 0.5], &[0.1, 0.7, 0.4], [0.0; 3]),
                        place(la, &[0.8, 2.1, 900.0], &[0.3, 0.9, 0.1], r),
                    ]);
                    let (s, t) = overlap_and_kinetic(&basis);
                    let alone = kinetic(&basis);
                    assert!(
                        t.iter()
                            .zip(&alone)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "<{la}|T|{lb}> at {r:?}"
                    );
                    let want = fb_s_overlap_host::overlap(&basis);
                    for (k, (got, want)) in s.iter().zip(&want).enumerate() {
                        assert!(
                            (got - want).abs() <= 1e-14,
                            "<{la}|{lb}> at {r:?} [{k}]: {got:e} vs {want:e}"
                        );
                    }
                }
            }
        }
    }

    /// A shell pair that is kept, whose tight primitives are past
    /// [`is_negligible`]'s bound and skipped (`mu r²` about 1800 for the
    /// tight pair, about 0.6 for the diffuse): the direct route agrees
    /// with the unscreened Cartesian one through i, to 1e-14 of the
    /// block's largest element.
    #[test]
    fn skipped_primitive_pairs_change_nothing() {
        let r = [0.96, -1.2, 1.28];
        let r2: f64 = r.iter().map(|x| x * x).sum();
        assert!(is_negligible(450.0, r2, 12) && !is_negligible(0.15, r2, 0));
        for la in 0..=6 {
            for lb in 0..=6 {
                let basis = BasisSet::new(vec![
                    place(lb, &[900.0, 0.3], &[0.2, 0.8], [0.0; 3]),
                    place(la, &[0.3, 900.0], &[0.8, 0.2], r),
                ]);
                assert_eq!(negligible_pairs(&basis), (0, 3), "<{la}|T|{lb}>");
                let (t, via) = (kinetic(&basis), kinetic_via_cartesian(&basis));
                let scale = via.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                for (k, (got, want)) in t.iter().zip(&via).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-14 * scale,
                        "<{la}|T|{lb}> [{k}]: {got:e} vs {want:e}"
                    );
                }
            }
        }
    }

    /// Just past [`is_negligible`]'s bound, where what it drops is
    /// largest, the full block of every pair of momenta through i is under
    /// 1e-15, by the Cartesian route, which screens nothing: contracted
    /// shells, off every axis; diffuse exponents, and tight ones, where the
    /// block's `mu` is large.
    #[test]
    fn negligible_pairs_are_below_1e_15() {
        for (ea, eb) in [([0.8, 2.1], [1.3, 0.5]), ([800.0, 2100.0], [1300.0, 500.0])] {
            let (a_min, b_min) = (ea[0], eb[1]);
            let mu = a_min * b_min / (a_min + b_min);
            for la in 0..=6 {
                for lb in 0..=6 {
                    let order = (la + lb) as usize;
                    let distance = ((40.0 + 4.0 * (order + 2) as f64) / mu).sqrt() * 1.0001;
                    let r = [0.48, -0.6, 0.64].map(|c| c * distance);
                    let basis = BasisSet::new(vec![
                        place(lb, &eb, &[0.7, 0.4], [0.0; 3]),
                        place(la, &ea, &[0.3, 0.9], r),
                    ]);
                    assert_eq!(negligible_pairs(&basis), (1, 3), "<{la}|T|{lb}>");
                    let (t, via) = (kinetic(&basis), kinetic_via_cartesian(&basis));
                    let (na, nb) = ((2 * la + 1) as usize, (2 * lb + 1) as usize);
                    let mut largest = 0.0_f64;
                    for row in nb..nb + na {
                        let cols = lower_index(row, 0)..lower_index(row, 0) + nb;
                        assert!(t[cols.clone()].iter().all(|&v| v == 0.0));
                        largest = via[cols].iter().fold(largest, |m, v| m.max(v.abs()));
                    }
                    assert!(largest <= 1e-15, "<{la}|T|{lb}>, mu {mu}: {largest:e}");
                }
            }
        }
    }
}
