//! The spherical AO nuclear-attraction matrix
//! `V[μ][ν] = -sum_C Z_C <μ| 1 / |r - C| |ν>` of a basis in the field of
//! point nuclei, computed on the host: plain Rust on rayon's threads, no
//! device and no kernel library.
//!
//! **The integral.** McMurchie-Davidson (Helgaker, "Molecular
//! Electronic-Structure Theory", 9.9). Two primitive Cartesian Gaussians of
//! exponents `a` and `b` on `A` and `B` multiply into Hermite Gaussians on
//! `P = (a A + b B) / p`, `p = a + b`, and against one point charge
//!
//! ```text
//! <a| 1/r_C |b> = 2 pi / p * exp(-mu |A - B|²)
//!                 * sum_tuv E^(ij)_t E^(kl)_u E^(mn)_v R_tuv(p, P - C)
//! ```
//!
//! with `mu = a b / p`, `E` the two-center Hermite expansion per axis and
//! `R_tuv` the Hermite Coulomb integrals from the Boys function
//! `F_n(p |P - C|²)` (the sibling DF metric's [`fb_df_metric_host::boys`],
//! the C++ tree's `eval_boys` ported).
//!
//! **Every nucleus in one table.** `E` depends on the primitive pair and not
//! on the nucleus, so the charges are summed into the Hermite table first:
//! `W_tuv = -sum_C Z_C R_tuv(p, P - C)`, one per primitive pair. The
//! Cartesian contraction with `E` then runs once per primitive pair, not
//! once per nucleus, and the nuclei cost a Boys evaluation and a recurrence
//! each.
//!
//! **Every primitive pair once.** `W` depends on the two exponents and the
//! two centers alone, not on the momenta or the coefficients, and a table
//! built to order `L` holds every lower one. A carried cc set is generally
//! contracted, stored as segments, so an atom's shells repeat their
//! exponents: on gly10 only 12 to 20% of the kept primitive pairs are
//! distinct. So a pair of centers first collects its distinct primitive
//! pairs, each to the highest `la + lb` a kept shell pair asks of it,
//! builds each one's `W` once, and then every shell pair's block reads
//! them.
//!
//! **Far nuclei skip the Boys function.** Once `x = p |P - C|²` is past
//! `df_metric`'s far-field bound, `exp(-x)` is nothing beside the Boys
//! values and the seeds of `R` have a closed form (see
//! `HermiteCoulomb::fill`): no `exp`, no `erf`, no series. On gly10 that is
//! 92% of the (primitive pair, nucleus) evaluations.
//!
//! **The layout.** `V` is over the `2l+1` real solid harmonics of each
//! shell, through the basis's own Cartesian-to-spherical block
//! ([`fb_basis::solid_harmonic_block`], scaled by `sqrt((2l-1)!!)` as
//! [`fb_basis::spherical_transform_matrix`] scales it). A contracted shell
//! pair's Cartesian block is summed over its primitive pairs, then turned
//! spherical once through the harmonic block's nonzeros alone. Coefficients
//! are taken as given: place the basis `primitive_normalized()`, as the
//! pipeline does, and the result is the tree's V, contractions never
//! renormalized.
//!
//! **The lower triangle alone.** `V` is symmetric, and [`potential_en`]
//! returns it packed: `(μ, ν)`, `ν <= μ`, at [`lower_index`], `n (n + 1) / 2`
//! values. Only the shell pairs `A >= B` are computed, one rayon task per
//! pair of centers, each into a buffer of its own that one pass then copies
//! into the triangle (`RAYON_NUM_THREADS` sets the thread count).
//!
//! **Negligible pairs.** The product of two Gaussians carries
//! `exp(-mu |A - B|²)`, and the Coulomb integral of what is left is at most
//! `2 pi / p` per unit charge, wherever the nuclei are. So a pair is left
//! zero when `mu |A - B|²` is at least `40 + 4 (la + lb)` plus the log of
//! `sum |Z| * 2 pi / p * |c_a c_b|` (see [`is_negligible`]): the overlap's
//! bound, raised by the largest the Coulomb factor can make it. The test is
//! per pair of shells, at the smallest exponents, and again per primitive
//! pair inside a pair that is kept.
//!
//! [`potential_en_via_cartesian`] is the textbook route, kept to check and
//! time against: Obara-Saika per nucleus, then the horizontal recurrence and
//! a dense sandwich `Uaᵀ V_ab Ub`, with no pair screened.

#![forbid(unsafe_code)]

// The hot loops are `f64::mul_add`, which is one instruction with the FMA
// target feature and a software call without it. `.cargo/config.toml` turns
// the feature on; this refuses a build that lost it (cargo run from another
// directory, or a RUSTFLAGS in the environment, which replaces the file's).
#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
compile_error!(
    "fb-v-potential-en-host needs the FMA target feature: build from its own directory, \
     whose .cargo/config.toml sets it, or add `-C target-feature=+fma` to RUSTFLAGS"
);

use std::f64::consts::PI;

use fb_basis::{BasisSet, Shell, cartesian_powers, solid_harmonic_block};
use fb_core_primitives::double_factorial_odd;
use fb_df_metric_host::{BOYS_MAX_ORDER, boys};
use rayon::prelude::*;

/// A point charge: a nucleus, at `position` in bohr, of charge `charge`
/// (its Z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCharge {
    /// The position, in bohr.
    pub position: [f64; 3],
    /// The charge, positive for a nucleus.
    pub charge: f64,
}

/// The nuclei of `molecule` as point charges, in its atoms' order.
#[must_use]
pub fn point_charges(molecule: &fb_molecule::Molecule) -> Vec<PointCharge> {
    molecule
        .atoms()
        .iter()
        .map(|atom| PointCharge {
            position: [atom.position.x, atom.position.y, atom.position.z],
            charge: f64::from(fb_molecule::nuclear_charge(atom)),
        })
        .collect()
}

/// Where `(μ, ν)`, `ν <= μ`, sits in the packed lower triangle: rows one
/// after another, row `μ` holding its `μ + 1` entries.
///
/// ```
/// use fb_v_potential_en_host::lower_index;
///
/// assert_eq!([lower_index(0, 0), lower_index(1, 0), lower_index(1, 1)], [0, 1, 2]);
/// assert_eq!(lower_index(2, 0), 3);
/// ```
#[must_use]
pub const fn lower_index(p: usize, q: usize) -> usize {
    p * (p + 1) / 2 + q
}

/// The squared length of `r`.
fn norm2(r: [f64; 3]) -> f64 {
    r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
}

/// `a - b`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

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
    /// The same block by its nonzeros, the columns one after another: the
    /// Cartesian function's index and its coefficient.
    terms: Vec<(usize, f64)>,
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
            for (index, &c) in column.iter().enumerate() {
                if c != 0.0 {
                    terms.push((index, c));
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
    fn column(&self, m: usize) -> &[(usize, f64)] {
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
///
/// # Panics
///
/// If a shell pair would ask the Boys function past
/// [`BOYS_MAX_ORDER`].
fn views(basis: &BasisSet) -> Vec<ShellView> {
    let l_max = basis
        .shells()
        .iter()
        .map(Shell::angular_momentum)
        .max()
        .unwrap_or(0);
    assert!(
        2 * l_max as usize <= BOYS_MAX_ORDER,
        "l = {l_max} is past the Boys tables' {BOYS_MAX_ORDER} / 2"
    );
    (0..=l_max).map(ShellView::new).collect()
}

/// The packed triangle `v` cut into the rows each of `bounds` owns, the
/// bounds consecutive from row 0: disjoint `&mut` slices, one per task.
fn split_rows(
    v: &mut [f64],
    bounds: impl Iterator<Item = std::ops::Range<usize>>,
) -> Vec<&mut [f64]> {
    let mut rows = Vec::new();
    let mut rest = v;
    for bound in bounds {
        let (mine, later) =
            rest.split_at_mut(lower_index(bound.end, 0) - lower_index(bound.start, 0));
        rows.push(mine);
        rest = later;
    }
    rows
}

/// Whether a pair of Gaussians of total momentum `order`, `r2` apart
/// squared, is negligible: `mu r²` at least `40 + 4 order + ln(max(1,
/// scale))`, `scale = sum |Z| * 2 pi / p * |c_a c_b|` the most the Coulomb
/// factor and the coefficients can raise the pair above its overlap's
/// `exp(-mu r²)`.
///
/// Taken at a shell pair's smallest exponents, it holds for every pair of
/// its primitives: a larger exponent raises `mu` and `p` both. The margin
/// `4 order` is the overlap's, for the polynomial the momenta put in front
/// of `exp(-mu r²)`; `negligible_pairs_are_below_1e_15` checks through i.
fn is_negligible(mu: f64, r2: f64, order: usize, scale: f64) -> bool {
    mu * r2 >= 40.0 + 4.0 * order as f64 + scale.max(1.0).ln()
}

/// One shell as the integral loops read it, the pair-independent factors
/// computed once.
struct ShellData {
    /// The angular momentum.
    l: usize,
    /// The center, in bohr.
    center: [f64; 3],
    /// The shell's first spherical function.
    row0: usize,
    /// Each primitive's exponent and coefficient.
    primitives: Vec<(f64, f64)>,
    /// Each primitive's exponent's place in its [`Center`]'s distinct
    /// exponents; set by [`centers`].
    slots: Vec<usize>,
    /// The smallest exponent.
    min_exponent: f64,
    /// The largest `|c|`.
    max_coefficient: f64,
}

impl ShellData {
    /// Shell `index` of `basis`.
    fn new(basis: &BasisSet, index: usize) -> ShellData {
        let shell = &basis.shells()[index];
        let c = shell.center();
        let primitives: Vec<(f64, f64)> = shell
            .primitives()
            .iter()
            .map(|p| (p.exponent, p.coefficient))
            .collect();
        ShellData {
            l: shell.angular_momentum() as usize,
            center: [c.x, c.y, c.z],
            row0: basis.ao_offset_sphe(index as u32) as usize,
            min_exponent: primitives.iter().fold(f64::INFINITY, |m, p| m.min(p.0)),
            max_coefficient: primitives.iter().fold(0.0, |m, p| m.max(p.1.abs())),
            slots: Vec::new(),
            primitives,
        }
    }

    /// The spherical function count.
    fn nsph(&self) -> usize {
        2 * self.l + 1
    }
}

/// Whether a pair of shells is negligible: [`is_negligible`] at their
/// smallest exponents and largest coefficients.
fn pair_is_negligible(a: &ShellData, b: &ShellData, r2: f64, charge_sum: f64) -> bool {
    let p = a.min_exponent + b.min_exponent;
    let mu = a.min_exponent * b.min_exponent / p;
    let scale = charge_sum * 2.0 * PI / p * a.max_coefficient * b.max_coefficient;
    is_negligible(mu, r2, a.l + b.l, scale)
}

/// `sum |Z|` over `charges`.
fn charge_sum(charges: &[PointCharge]) -> f64 {
    charges.iter().map(|c| c.charge.abs()).sum()
}

/// A run of consecutive shells on one center, an atom's: its shells, its
/// distinct exponents and where it sits.
struct Center {
    /// The shells.
    shells: std::ops::Range<usize>,
    /// The exponents its shells hold, each once (equal to the bit).
    exponents: Vec<f64>,
    /// The position, in bohr.
    at: [f64; 3],
}

/// The shells of `data` grouped into [`Center`]s, in order, each shell's
/// [`ShellData::slots`] set to its center's exponents.
fn centers(data: &mut [ShellData]) -> Vec<Center> {
    let mut out: Vec<Center> = Vec::new();
    for (index, shell) in data.iter_mut().enumerate() {
        let at = shell.center;
        match out.last_mut() {
            Some(last) if last.at == at => last.shells.end = index + 1,
            _ => out.push(Center {
                shells: index..index + 1,
                exponents: Vec::new(),
                at,
            }),
        }
        let center = out.last_mut().expect("just pushed");
        shell.slots = shell
            .primitives
            .iter()
            .map(|&(a, _)| {
                center
                    .exponents
                    .iter()
                    .position(|&e| e.to_bits() == a.to_bits())
                    .unwrap_or_else(|| {
                        center.exponents.push(a);
                        center.exponents.len() - 1
                    })
            })
            .collect();
    }
    out
}

/// Whether the Boys function's `exp(-x)` is nothing beside what is kept, at
/// `x = p |P - C|²` and highest order `l`: `df_metric`'s far-field bound,
/// where the dropped term is under 1e-17 of `F_n` for every order to 20.
fn is_far(x: f64, l: usize) -> bool {
    x >= 40.0 + 4.0 * l as f64
}

/// The Hermite Coulomb integrals `R^n_tuv` of total order up to `l` at
/// exponent `p` over the separation `r = P - C`, built from
/// `R^n_000 = (-2p)^n F_n(p r²)` by
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
    ///
    /// A far separation ([`is_far`]) takes the seeds in closed form, with
    /// no Boys function: there `F_n(x) = (2n-1)!! / (2x)^n sqrt(pi / x) / 2`,
    /// so `R^n_000 = sqrt(pi / p) / 2 * (-1)^n (2n-1)!! / r^(2n+1)`.
    fn fill(&mut self, l: usize, p: f64, r: [f64; 3]) {
        let d = l + 1;
        self.d = d;
        let r2 = norm2(r);
        if is_far(p * r2, l) {
            let inverse = 1.0 / r2;
            let mut value = 0.5 * (PI / p).sqrt() * inverse.sqrt();
            for n in 0..=l {
                self.r[n * d * d * d] = value;
                value *= -((2 * n + 1) as f64) * inverse;
            }
        } else {
            boys(p * r2, &mut self.f[..=l]);
            let mut power = 1.0;
            for n in 0..=l {
                self.r[n * d * d * d] = power * self.f[n];
                power *= -2.0 * p;
            }
        }
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

/// The two-center Hermite expansion of one axis, `E^(ij)_t` of
/// `x_A^i x_B^j exp(-a x_A² - b x_B²) = exp(-mu X²) sum_t E^(ij)_t Λ_t`,
/// `i <= la`, `j <= lb`, `t <= i + j`, into `e` at
/// `(i * (lb + 1) + j) * d + t`, `d = la + lb + 1`: `E^(00)_0 = 1`, and
/// `E^(i+1,j)_t = E^(ij)_(t-1) / (2p) + X_PA E^(ij)_t + (t+1) E^(ij)_(t+1)`
/// and its `j` twin with `X_PB`.
fn hermite_expansion(la: usize, lb: usize, p: f64, pa: f64, pb: f64, e: &mut [f64]) {
    let d = la + lb + 1;
    let db = lb + 1;
    let half = 0.5 / p;
    let e = &mut e[..(la + 1) * db * d];
    e.fill(0.0);
    e[0] = 1.0;
    let step = |e: &mut [f64], from: usize, to: usize, top: usize, x: f64| {
        for t in 0..=top + 1 {
            let below = if t > 0 { e[from + t - 1] } else { 0.0 };
            let above = if t < top { e[from + t + 1] } else { 0.0 };
            e[to + t] = half.mul_add(below, x.mul_add(e[from + t], (t + 1) as f64 * above));
        }
    };
    for i in 0..la {
        step(e, i * db * d, (i + 1) * db * d, i, pa);
    }
    for i in 0..=la {
        for j in 0..lb {
            step(e, (i * db + j) * d, (i * db + j + 1) * d, i + j, pb);
        }
    }
}

/// One distinct primitive pair of a pair of centers, its exponent pair
/// `(a, b)`: what every shell pair holding it shares.
#[derive(Clone, Copy, Default)]
struct PrimitivePair {
    /// The highest `la + lb` a shell pair asks of it, or `None` if none
    /// does.
    order: Option<usize>,
    /// `p = a + b`.
    p: f64,
    /// `P`, in bohr.
    at: [f64; 3],
    /// `2 pi / p exp(-mu |A - B|²)`.
    prefactor: f64,
    /// Where its `W` starts in [`Scratch::w`]; its side is `order + 1`.
    offset: usize,
}

/// What one pair of centers is built in, one per rayon job, sized for the
/// basis's largest momentum except the shared tables, which grow.
struct Scratch {
    /// The three axes' Hermite expansions, each `(l_max + 1)² (2 l_max + 1)`.
    e: [Vec<f64>; 3],
    /// One nucleus's Hermite Coulomb integrals.
    table: HermiteCoulomb,
    /// The pair of centers' distinct primitive pairs, `a`'s slot by `b`'s.
    pairs: Vec<PrimitivePair>,
    /// Every distinct primitive pair's `W_tuv = -sum_C Z_C R_tuv`,
    /// `t + u + v <= order`, at `offset + (t * d + u) * d + v`,
    /// `d = order + 1`.
    w: Vec<f64>,
    /// The contracted Cartesian block, `ncart_a` rows by `ncart_b` columns.
    cartesian: Vec<f64>,
}

impl Scratch {
    /// Scratch for shells up to momentum `l_max`.
    fn new(l_max: usize) -> Scratch {
        let d = 2 * l_max + 1;
        let side = l_max + 1;
        let ncart = side * (side + 1) / 2;
        let axis = || vec![0.0; side * side * d];
        Scratch {
            e: [axis(), axis(), axis()],
            table: HermiteCoulomb::new(2 * l_max),
            pairs: Vec::new(),
            w: Vec::new(),
            cartesian: vec![0.0; ncart * ncart],
        }
    }
}

/// The shell pairs of centers `ca >= cb` in the order [`potential_en`]
/// lays out their blocks: `a` over `ca`'s shells, `b` over `cb`'s, and of a
/// center with itself only `b <= a`.
fn shell_pairs(centers: &[Center], ca: usize, cb: usize) -> impl Iterator<Item = (usize, usize)> {
    let (a_shells, b_shells) = (centers[ca].shells.clone(), centers[cb].shells.clone());
    a_shells.flat_map(move |sa| {
        let last = if ca == cb { sa + 1 } else { b_shells.end };
        (b_shells.start..last).map(move |sb| (sa, sb))
    })
}

/// Every spherical block of the shell pairs of centers `ca >= cb`, one
/// after another in [`shell_pairs`]' order, `nsph_a × nsph_b` each at
/// `ia * nsph_b + ib`; a negligible pair's block is zero.
///
/// First the distinct primitive pairs the kept shell pairs ask for, each
/// to the highest order one asks, then each one's `W` (the module's "Every
/// primitive pair once"), then the blocks from them.
#[allow(clippy::too_many_arguments)]
fn center_pair(
    data: &[ShellData],
    views: &[ShellView],
    centers: &[Center],
    ca: usize,
    cb: usize,
    charges: &[PointCharge],
    charge_sum: f64,
    scratch: &mut Scratch,
) -> Vec<f64> {
    let Scratch {
        e,
        table,
        pairs,
        w,
        cartesian,
    } = scratch;
    let (center_a, center_b) = (&centers[ca], &centers[cb]);
    let (at_a, at_b) = (center_a.at, center_b.at);
    let r2 = norm2(sub(at_a, at_b));
    let nb = center_b.exponents.len();
    pairs.clear();
    pairs.resize(center_a.exponents.len() * nb, PrimitivePair::default());

    // Which primitive pairs the kept shell pairs ask for, and to what order.
    for (sa, sb) in shell_pairs(centers, ca, cb) {
        let (a, b) = (&data[sa], &data[sb]);
        if pair_is_negligible(a, b, r2, charge_sum) {
            continue;
        }
        let order = a.l + b.l;
        for (&(ax, c_a), &slot_a) in a.primitives.iter().zip(&a.slots) {
            for (&(bx, c_b), &slot_b) in b.primitives.iter().zip(&b.slots) {
                let p = ax + bx;
                let scale = charge_sum * 2.0 * PI / p * (c_a * c_b).abs();
                if !is_negligible(ax * bx / p, r2, order, scale) {
                    let pair = &mut pairs[slot_a * nb + slot_b];
                    pair.order = Some(pair.order.map_or(order, |o| o.max(order)));
                }
            }
        }
    }

    // Each one's W, every nucleus summed in.
    w.clear();
    for (slot, pair) in pairs.iter_mut().enumerate() {
        let Some(order) = pair.order else { continue };
        let (ax, bx) = (center_a.exponents[slot / nb], center_b.exponents[slot % nb]);
        let p = ax + bx;
        pair.p = p;
        pair.at = [0, 1, 2].map(|k| (ax * at_a[k] + bx * at_b[k]) / p);
        pair.prefactor = 2.0 * PI / p * (-ax * bx / p * r2).exp();
        pair.offset = w.len();
        let d = order + 1;
        w.resize(w.len() + d * d * d, 0.0);
        let w = &mut w[pair.offset..];
        for charge in charges {
            table.fill(order, p, sub(pair.at, charge.position));
            let z = -charge.charge;
            for t in 0..=order {
                for u in 0..=order - t {
                    let row = (t * d + u) * d;
                    for v in 0..=order - t - u {
                        w[row + v] = z.mul_add(table.get(t, u, v), w[row + v]);
                    }
                }
            }
        }
    }

    // The blocks.
    let mut out = Vec::new();
    for (sa, sb) in shell_pairs(centers, ca, cb) {
        let (a, b) = (&data[sa], &data[sb]);
        let (va, vb) = (&views[a.l], &views[b.l]);
        let (nsa, nsb) = (va.nsph(), vb.nsph());
        let start = out.len();
        out.resize(start + nsa * nsb, 0.0);
        if pair_is_negligible(a, b, r2, charge_sum) {
            continue;
        }
        let order = a.l + b.l;
        let db = b.l + 1;
        let de = order + 1;
        let (nca, ncb) = (va.ncart(), vb.ncart());
        let cartesian = &mut cartesian[..nca * ncb];
        cartesian.fill(0.0);
        for (&(ax, c_a), &slot_a) in a.primitives.iter().zip(&a.slots) {
            for (&(bx, c_b), &slot_b) in b.primitives.iter().zip(&b.slots) {
                let p = ax + bx;
                let coefficient = c_a * c_b;
                let scale = charge_sum * 2.0 * PI / p * coefficient.abs();
                if is_negligible(ax * bx / p, r2, order, scale) {
                    continue;
                }
                let pair = &pairs[slot_a * nb + slot_b];
                let side = pair.order.expect("asked for above") + 1;
                let w = &w[pair.offset..pair.offset + side * side * side];
                let weight = coefficient * pair.prefactor;
                if order == 0 {
                    cartesian[0] = weight.mul_add(w[0], cartesian[0]);
                    continue;
                }
                for (k, axis) in e.iter_mut().enumerate() {
                    let (pa, pb) = (pair.at[k] - at_a[k], pair.at[k] - at_b[k]);
                    hermite_expansion(a.l, b.l, p, pa, pb, axis);
                }
                let [ex, ey, ez] = &*e;
                for (ia, &[i, j, k]) in va.powers.iter().enumerate() {
                    for (ib, &[l, m, n]) in vb.powers.iter().enumerate() {
                        let (x, y, z) = ((i * db + l) * de, (j * db + m) * de, (k * db + n) * de);
                        let mut sum = 0.0;
                        for t in 0..=i + l {
                            let wx = ex[x + t];
                            for u in 0..=j + m {
                                let wxy = wx * ey[y + u];
                                let row = (t * side + u) * side;
                                for v in 0..=k + n {
                                    sum = (wxy * ez[z + v]).mul_add(w[row + v], sum);
                                }
                            }
                        }
                        let to = &mut cartesian[ia * ncb + ib];
                        *to = weight.mul_add(sum, *to);
                    }
                }
            }
        }
        let block = &mut out[start..];
        for (ia, row) in block.chunks_exact_mut(nsb).enumerate() {
            let column_a = va.column(ia);
            for (jb, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for &(pa, c_a) in column_a {
                    let mut inner = 0.0;
                    for &(pb, c_b) in vb.column(jb) {
                        inner = c_b.mul_add(cartesian[pa * ncb + pb], inner);
                    }
                    sum = c_a.mul_add(inner, sum);
                }
                *value = sum;
            }
        }
    }
    out
}

/// The lower triangle of the nuclear attraction
/// `V[μ][ν] = -sum_C Z_C <μ| 1 / |r - C| |ν>` of `basis` in the field of
/// `charges` over its spherical layout, packed: `n (n + 1) / 2` values for
/// `n = nao_sphe`, `(μ, ν)`, `ν <= μ`, at [`lower_index`]`(μ, ν)`. `V` is
/// symmetric, so the upper triangle is never computed and never stored.
///
/// Every nucleus is summed into one Hermite table per distinct primitive
/// pair of a pair of centers (the module's "Every nucleus in one table" and
/// "Every primitive pair once"); a negligible pair is left zero. The
/// coefficients are used as `basis` holds them; place it
/// `primitive_normalized()`.
///
/// One rayon task per pair of centers, since the shells sharing a
/// primitive pair are those of the two centers. Each writes its blocks to
/// a buffer of its own; one pass then copies them into the triangle.
///
/// # Panics
///
/// If the basis reaches past `l = 10`, where the Boys tables stop.
#[must_use]
pub fn potential_en(basis: &BasisSet, charges: &[PointCharge]) -> Vec<f64> {
    let n = basis.nao_sphe() as usize;
    let views = views(basis);
    let mut data: Vec<ShellData> = (0..basis.shells().len())
        .map(|index| ShellData::new(basis, index))
        .collect();
    let centers = centers(&mut data);
    let charge_sum = charge_sum(charges);
    let l_max = views.len() - 1;
    let center_pairs: Vec<(usize, usize)> = (0..centers.len())
        .flat_map(|ca| (0..=ca).map(move |cb| (ca, cb)))
        .collect();
    let blocks: Vec<Vec<f64>> = center_pairs
        .par_iter()
        .map_init(
            || Scratch::new(l_max),
            |scratch, &(ca, cb)| {
                center_pair(
                    &data, &views, &centers, ca, cb, charges, charge_sum, scratch,
                )
            },
        )
        .collect();

    let mut v = vec![0.0; lower_index(n, 0)];
    for (&(ca, cb), blocks) in center_pairs.iter().zip(&blocks) {
        let mut from = 0;
        for (sa, sb) in shell_pairs(&centers, ca, cb) {
            let (a, b) = (&data[sa], &data[sb]);
            let nsb = b.nsph();
            for ia in 0..a.nsph() {
                // Shells are laid out in order, so an earlier shell's
                // columns are all below the diagonal.
                let cols = if sa == sb { ia + 1 } else { nsb };
                let at = lower_index(a.row0 + ia, b.row0);
                v[at..at + cols].copy_from_slice(&blocks[from + ia * nsb..from + ia * nsb + cols]);
            }
            from += a.nsph() * nsb;
        }
    }
    v
}

/// How many of `basis`'s shell pairs `A >= B` [`potential_en`] leaves zero
/// as negligible in the field of `charges`, and how many pairs there are.
#[must_use]
pub fn negligible_pairs(basis: &BasisSet, charges: &[PointCharge]) -> (usize, usize) {
    let data: Vec<ShellData> = (0..basis.shells().len())
        .map(|index| ShellData::new(basis, index))
        .collect();
    let charge_sum = charge_sum(charges);
    let negligible: usize = data
        .par_iter()
        .enumerate()
        .map(|(sa, a)| {
            data[..sa]
                .iter()
                .filter(|b| pair_is_negligible(a, b, norm2(sub(a.center, b.center)), charge_sum))
                .count()
        })
        .sum();
    (negligible, data.len() * (data.len() + 1) / 2)
}

/// The index of `x^i y^j z^k` in a dense cube of side `d`.
fn cube(e: [usize; 3], d: usize) -> usize {
    (e[0] * d + e[1]) * d + e[2]
}

/// The contracted Cartesian block `<a|V|b>` of two shells, `ncart_a` rows
/// by `ncart_b` columns at `ia * ncart_b + ib`, by Obara-Saika and the
/// horizontal recurrence.
///
/// Per primitive pair and nucleus, the vertical recurrence builds
/// `[e|0]^(m)`, `|e| <= la + lb`, from
/// `[0|0]^(m) = 2 pi / p exp(-mu |A - B|²) F_m(p |P - C|²)` by
///
/// ```text
/// [e+1_i|0]^(m) = PA_i [e]^(m) - PC_i [e]^(m+1)
///                 + e_i / (2p) ([e-1_i]^(m) - [e-1_i]^(m+1))
/// ```
///
/// The `m = 0` values are summed under `-Z c_a c_b` over nuclei and primitive pairs. The
/// horizontal recurrence `[a|b+1_i] = [a+1_i|b] + AB_i [a|b]` holds no
/// exponent, so it runs once on that sum.
fn cartesian_block(
    a: &Shell,
    va: &ShellView,
    b: &Shell,
    vb: &ShellView,
    charges: &[PointCharge],
) -> Vec<f64> {
    let (ca, cb) = (a.center(), b.center());
    let (at_a, at_b) = ([ca.x, ca.y, ca.z], [cb.x, cb.y, cb.z]);
    let ab = sub(at_a, at_b);
    let order = va.l + vb.l;
    let d = order + 1;
    let volume = d * d * d;
    // `[e|0]^(m)` at `cube(e) * d + m`, and its sum at `cube(e)`.
    let mut vrr = vec![0.0; volume * d];
    let mut sum = vec![0.0; volume];
    let mut f = vec![0.0; d];
    let shells_by_order: Vec<Vec<[usize; 3]>> = (0..=order)
        .map(|l| {
            cartesian_powers(l as i32)
                .iter()
                .map(|p| p.map(|c| c as usize))
                .collect()
        })
        .collect();
    for pa in a.primitives() {
        for pb in b.primitives() {
            let (ax, bx) = (pa.exponent, pb.exponent);
            let p = ax + bx;
            let at = [0, 1, 2].map(|k| (ax * at_a[k] + bx * at_b[k]) / p);
            let to_a = sub(at, at_a);
            let prefactor = 2.0 * PI / p * (-ax * bx / p * norm2(ab)).exp();
            let half = 0.5 / p;
            for charge in charges {
                let pc = sub(at, charge.position);
                boys(p * norm2(pc), &mut f);
                for m in 0..d {
                    vrr[m] = prefactor * f[m];
                }
                for (l, powers) in shells_by_order.iter().enumerate().skip(1) {
                    for &e in powers {
                        let i = (0..3).find(|&k| e[k] > 0).expect("l > 0");
                        let mut down = e;
                        down[i] -= 1;
                        let (here, one) = (cube(e, d) * d, cube(down, d) * d);
                        let two = (down[i] > 0).then(|| {
                            let mut two = down;
                            two[i] -= 1;
                            cube(two, d) * d
                        });
                        let factor = down[i] as f64 * half;
                        for m in 0..d - l {
                            let mut value = to_a[i] * vrr[one + m] - pc[i] * vrr[one + m + 1];
                            if let Some(two) = two {
                                value += factor * (vrr[two + m] - vrr[two + m + 1]);
                            }
                            vrr[here + m] = value;
                        }
                    }
                }
                let weight = -charge.charge * pa.coefficient * pb.coefficient;
                for powers in &shells_by_order {
                    for &e in powers {
                        sum[cube(e, d)] += weight * vrr[cube(e, d) * d];
                    }
                }
            }
        }
    }
    // The horizontal recurrence: `hrr[cube(b, db) * volume + cube(a, d)]`
    // for `|a| <= order - |b|`, by increasing `|b|`.
    let db = vb.l + 1;
    let mut hrr = vec![0.0; db * db * db * volume];
    hrr[..volume].copy_from_slice(&sum);
    for (lb, powers) in shells_by_order.iter().enumerate().take(vb.l + 1).skip(1) {
        for &bb in powers {
            let i = (0..3).find(|&k| bb[k] > 0).expect("l > 0");
            let mut down = bb;
            down[i] -= 1;
            let (here, from) = (cube(bb, db) * volume, cube(down, db) * volume);
            for powers_a in &shells_by_order[..=order - lb] {
                for &aa in powers_a {
                    let mut up = aa;
                    up[i] += 1;
                    hrr[here + cube(aa, d)] =
                        hrr[from + cube(up, d)] + ab[i] * hrr[from + cube(aa, d)];
                }
            }
        }
    }
    let mut block = vec![0.0; va.ncart() * vb.ncart()];
    for (ia, &pa) in va.powers.iter().enumerate() {
        for (ib, &pb) in vb.powers.iter().enumerate() {
            block[ia * vb.ncart() + ib] = hrr[cube(pb, db) * volume + cube(pa, d)];
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

/// [`potential_en`] by the textbook route, kept to check and time it
/// against: every Cartesian `<a|V|b>` of a shell pair by Obara-Saika per
/// nucleus and the horizontal recurrence, then `Uaᵀ <a|V|b> Ub`, with no
/// pair screened. The same packed triangle to rounding. One rayon task per
/// shell.
///
/// # Panics
///
/// If the basis reaches past `l = 10`, where the Boys tables stop.
#[must_use]
pub fn potential_en_via_cartesian(basis: &BasisSet, charges: &[PointCharge]) -> Vec<f64> {
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
    let mut v = vec![0.0; lower_index(n, 0)];
    let rows = split_rows(&mut v, (0..shells.len()).map(|i| row0(i)..row0(i + 1)));

    rows.into_par_iter().enumerate().for_each(|(sa, rows)| {
        let a = &shells[sa];
        let va = &views[a.angular_momentum() as usize];
        let start = lower_index(row0(sa), 0);
        for (sb, b) in shells[..=sa].iter().enumerate() {
            let vb = &views[b.angular_momentum() as usize];
            let block = to_spherical(&cartesian_block(a, va, b, vb, charges), va, vb);
            let nsb = vb.nsph();
            for ia in 0..va.nsph() {
                let cols = if sa == sb { ia + 1 } else { nsb };
                let at = lower_index(row0(sa) + ia, row0(sb)) - start;
                rows[at..at + cols].copy_from_slice(&block[ia * nsb..ia * nsb + cols]);
            }
        }
    });
    v
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

    /// Three nuclei off every axis: on neither shell, between them, and on
    /// one.
    fn three_nuclei() -> Vec<PointCharge> {
        vec![
            PointCharge {
                position: [0.9, -0.4, 1.3],
                charge: 8.0,
            },
            PointCharge {
                position: [0.2, -0.3, 0.3],
                charge: 1.0,
            },
            PointCharge {
                position: [0.0; 3],
                charge: 6.0,
            },
        ]
    }

    /// Two normalized s primitives against one nucleus:
    /// `V = -Z N_a N_b 2 pi / p exp(-mu R²) F_0(p |P - C|²)`, on one center
    /// and apart, by both routes.
    #[test]
    fn two_s_primitives_have_the_closed_form() {
        let (a, b) = (0.9, 2.3);
        let nucleus = PointCharge {
            position: [0.3, -0.5, 0.8],
            charge: 3.0,
        };
        let norm = |x: f64| (2.0 * x / PI).powf(0.75);
        for z in [0.0, 1.4] {
            let basis = BasisSet::new(vec![
                place(0, &[a], &[1.0], [0.0; 3]),
                place(0, &[b], &[1.0], [0.0, 0.0, z]),
            ]);
            let v = potential_en(&basis, &[nucleus]);
            let via = potential_en_via_cartesian(&basis, &[nucleus]);
            let p = a + b;
            let at = [0.0, 0.0, b * z / p];
            let mut f = [0.0];
            boys(p * norm2(sub(at, nucleus.position)), &mut f);
            let want = -3.0 * norm(a) * norm(b) * 2.0 * PI / p * (-a * b / p * z * z).exp() * f[0];
            assert_eq!(v.len(), 3);
            assert!(
                (v[1] - want).abs() <= 1e-15 * want.abs(),
                "{} vs {want}",
                v[1]
            );
            assert!(
                (via[1] - want).abs() <= 1e-15 * want.abs(),
                "{} vs {want}",
                via[1]
            );
        }
    }

    /// A nucleus far from a normalized primitive of every momentum through
    /// i is a point charge seen from outside a unit-norm cloud: the diagonal
    /// tends to `-Z / |C - A|` and the off-diagonal to 0, to the cloud's
    /// quadrupole, which falls as `|C - A|^-3`.
    #[test]
    fn a_far_nucleus_sees_a_unit_charge() {
        let distance = 400.0;
        let nucleus = PointCharge {
            position: [0.48 * distance, -0.6 * distance, 0.64 * distance],
            charge: 2.0,
        };
        for l in 0..=6 {
            let basis = BasisSet::new(vec![place(l, &[1.7], &[1.0], [0.0; 3])]);
            for (route, v) in [
                ("spherical", potential_en(&basis, &[nucleus])),
                (
                    "via cartesian",
                    potential_en_via_cartesian(&basis, &[nucleus]),
                ),
            ] {
                for m in 0..(2 * l + 1) as usize {
                    for k in 0..m {
                        assert!(v[lower_index(m, k)].abs() <= 1e-7, "{route}, l {l}");
                    }
                    let diag = v[lower_index(m, m)];
                    assert!(
                        (diag + 2.0 / distance).abs() <= 1e-7,
                        "{route}, l {l}, m {m}: {diag}"
                    );
                }
            }
        }
    }

    /// The two routes agree on contracted shells of every pair of momenta
    /// through i, off every axis, in the field of three nuclei, to 1e-13 of
    /// the largest element.
    #[test]
    fn the_routes_agree_through_i() {
        let charges = three_nuclei();
        for la in 0..=6 {
            for lb in 0..=6 {
                let basis = BasisSet::new(vec![
                    place(lb, &[1.3, 0.5], &[0.7, 0.4], [0.0; 3]),
                    place(la, &[0.8, 2.1], &[0.3, 0.9], [0.48, -0.6, 0.64]),
                ]);
                let (v, via) = (
                    potential_en(&basis, &charges),
                    potential_en_via_cartesian(&basis, &charges),
                );
                let scale = via.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                for (k, (got, want)) in v.iter().zip(&via).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-13 * scale,
                        "<{la}|V|{lb}> [{k}]: {got:e} vs {want:e}"
                    );
                }
            }
        }
    }

    /// Just past [`pair_is_negligible`]'s bound, the full block of every
    /// pair of momenta through i is under 1e-15: contracted shells, off
    /// every axis, nuclei on both shells and between them, with no
    /// screening.
    #[test]
    fn negligible_pairs_are_below_1e_15() {
        let (ea, eb) = ([0.8, 2.1], [1.3, 0.5]);
        let (ca, cb) = ([0.3, 0.9], [0.7, 0.4]);
        for la in 0..=6 {
            for lb in 0..=6 {
                let unit = [0.48, -0.6, 0.64];
                let probe = BasisSet::new(vec![
                    place(lb, &eb, &cb, [0.0; 3]),
                    place(la, &ea, &ca, unit),
                ]);
                let (b, a) = (ShellData::new(&probe, 0), ShellData::new(&probe, 1));
                let charge_sum = 8.0 + 1.0 + 6.0;
                // The distance where the bound is met, by bisection.
                let (mut near, mut far) = (0.0, 1e3);
                for _ in 0..200 {
                    let mid = 0.5 * (near + far);
                    if pair_is_negligible(&a, &b, mid * mid, charge_sum) {
                        far = mid;
                    } else {
                        near = mid;
                    }
                }
                let distance = far * 1.0001;
                let r = unit.map(|c| c * distance);
                let charges = vec![
                    PointCharge {
                        position: r,
                        charge: 8.0,
                    },
                    PointCharge {
                        position: r.map(|c| 0.5 * c),
                        charge: 1.0,
                    },
                    PointCharge {
                        position: [0.0; 3],
                        charge: 6.0,
                    },
                ];
                let basis =
                    BasisSet::new(vec![place(lb, &eb, &cb, [0.0; 3]), place(la, &ea, &ca, r)]);
                assert_eq!(negligible_pairs(&basis, &charges), (1, 3), "<{la}|{lb}>");
                let v = potential_en(&basis, &charges);
                let nb = (2 * lb + 1) as usize;
                assert!(
                    v[lower_index(nb, 0)..lower_index(nb, 0) + nb]
                        .iter()
                        .all(|&x| x == 0.0)
                );
                let full = potential_en_via_cartesian(&basis, &charges);
                let largest = (0..(2 * la + 1) as usize)
                    .flat_map(|ia| (0..nb).map(move |jb| (ia, jb)))
                    .map(|(ia, jb)| full[lower_index(nb + ia, jb)].abs())
                    .fold(0.0_f64, f64::max);
                assert!(largest <= 1e-15, "<{la}|V|{lb}>: {largest:e}");
            }
        }
    }
}
