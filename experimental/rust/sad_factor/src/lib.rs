//! The SAD guess as a factor: `L`, `n_ao × rank`, with `D_SAD = 2 L Lᵀ`.
//!
//! A J/K build that is given the occupied orbitals C of `D = 2 C Cᵀ` builds
//! K at `n_occ / n_ao` of what it costs from D. The SAD density has no such
//! C: it is not the density of any orbitals of the molecule. But it has a
//! factor, and a J/K build that reads only `C Cᵀ` takes it in C's place.
//!
//! **Per element, never the molecule's matrix.** D_SAD is each atom's stamp
//! `D_A` on its diagonal block and zero between atoms. A stamp is an atomic
//! SCF's density, `Σ_i n_i c_i c_iᵀ` over its occupied orbitals with
//! `n_i >= 0`, so it is positive semidefinite, and its rank is the atom's
//! count of occupied orbitals, not its AO count. Its eigendecomposition
//! gives the factor:
//!
//! ```text
//! D_A = V Λ Vᵀ   →   L_A = V[:, λ > tol] · sqrt(λ / 2)
//! ```
//!
//! and L is every atom's `L_A` on that atom's rows, each in columns of its
//! own, atom after atom. One decomposition per distinct (Z, shells), as one
//! SCF per distinct (Z, shells) in `fb_sad`.
//!
//! **The columns are not the atomic orbitals.** Those are orthonormal in
//! the atom's overlap, the eigenvectors in the plain dot product, so the
//! two span the same space in different bases. Only `L Lᵀ` is read.
//!
//! **The cut.** A kept eigenvalue is past [`RANK_TOLERANCE`] of the
//! largest. The nonzero eigenvalues of `C n Cᵀ` are those of `√n CᵀC √n`,
//! bounded below by `n_min / λ_max(S)`, so they stay orders of magnitude
//! above the rounding the null space comes out as; [`StampFactor`] carries
//! both sides of the gap so a caller can see it.

#![forbid(unsafe_code)]

use fb_atom::Atom;
use fb_basis::BasisSet;
use fb_core_error_handling::Result;
use fb_sad::{SadStampLookup, distinct_stamps, stamp_density};

/// An eigenvalue of a stamp is kept when it is past this fraction of the
/// stamp's largest.
pub const RANK_TOLERANCE: f64 = 1e-10;

/// The eigenvalues, ascending, and eigenvectors (column j for value j,
/// column-major) of the symmetric `n × n` column-major `a`, by cyclic
/// Jacobi: `fb_sad`'s solver for its l blocks, which that crate keeps
/// private. A stamp is at most an atom's AOs on a side.
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i + n * i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for j in 0..n {
            for i in 0..j {
                off += a[i + n * j] * a[i + n * j];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p + n * q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q + n * q] - a[p + n * p]) / (2.0 * apq);
                let t = 1.0_f64.copysign(theta) / (theta.abs() + theta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k + n * p], a[k + n * q]);
                    a[k + n * p] = c * akp - s * akq;
                    a[k + n * q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p + n * k], a[q + n * k]);
                    a[p + n * k] = c * apk - s * aqk;
                    a[q + n * k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k + n * p], v[k + n * q]);
                    v[k + n * p] = c * vkp - s * vkq;
                    v[k + n * q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x + n * x].total_cmp(&a[y + n * y]));
    let values = order.iter().map(|&o| a[o + n * o]).collect();
    let mut vectors = vec![0.0; n * n];
    for (j, &o) in order.iter().enumerate() {
        vectors[n * j..n * (j + 1)].copy_from_slice(&v[n * o..n * (o + 1)]);
    }
    (values, vectors)
}

/// One stamp's factor: `L_A`, with `D_A = 2 L_A L_Aᵀ`.
pub struct StampFactor {
    /// The stamp's side, the atom's spherical AO count.
    pub rows: usize,
    /// The columns kept: the stamp's rank.
    pub rank: usize,
    /// `L_A`, `rows × rank` column-major, the largest eigenvalue's column
    /// first.
    pub columns: Vec<f64>,
    /// The smallest eigenvalue kept; 0 if none was.
    pub smallest_kept: f64,
    /// The largest `|λ|` dropped; 0 if none was.
    pub largest_dropped: f64,
}

/// The factor of the `n × n` column-major stamp `density`:
/// `V[:, λ > tol] · sqrt(λ / 2)` of its eigendecomposition, `tol`
/// [`RANK_TOLERANCE`] of the largest eigenvalue.
///
/// # Panics
/// If `density` is not `n × n`.
pub fn stamp_factor(density: &[f64], n: usize) -> StampFactor {
    assert_eq!(density.len(), n * n, "stamp_factor: the stamp is n × n");
    let (values, vectors) = symmetric_eigen(density.to_vec(), n);
    let cut = RANK_TOLERANCE * values.last().copied().unwrap_or(0.0).max(0.0);
    // Ascending, so the kept ones are the tail.
    let first_kept = values.partition_point(|&value| value <= cut);
    let rank = n - first_kept;
    let mut columns = Vec::with_capacity(n * rank);
    for j in (first_kept..n).rev() {
        let scale = (values[j] / 2.0).sqrt();
        columns.extend(vectors[n * j..n * (j + 1)].iter().map(|v| v * scale));
    }
    StampFactor {
        rows: n,
        rank,
        columns,
        smallest_kept: values.get(first_kept).copied().unwrap_or(0.0),
        largest_dropped: values[..first_kept]
            .iter()
            .fold(0.0, |m, value| m.max(value.abs())),
    }
}

/// One distinct (Z, shells) of a molecule: its factor and how many atoms
/// wear it.
pub struct ElementFactor {
    /// The atomic number.
    pub z: i32,
    /// The atoms that wear this stamp.
    pub atoms: usize,
    /// The stamp's factor.
    pub factor: StampFactor,
}

/// The SAD guess's factor: `L`, with `D_SAD = 2 L Lᵀ`.
pub struct SadFactor {
    /// `basis.nao_sphe()`.
    pub n_ao: usize,
    /// L's columns: the sum of the atoms' stamp ranks.
    pub rank: usize,
    /// `L`, `n_ao × rank` column-major. Each atom's columns are nonzero on
    /// that atom's rows alone, atom after atom in the basis's order.
    pub columns: Vec<f64>,
    /// Each distinct (Z, shells), in order of its first atom.
    pub elements: Vec<ElementFactor>,
}

/// The factor of `fb_sad::sad_density(basis, atoms, baked)`, from one
/// eigendecomposition per distinct (Z, shells) and never the `n_ao × n_ao`
/// matrix. `basis`, `atoms` and `baked` are that function's.
///
/// # Errors
/// `fb_sad::sad_density`'s.
pub fn sad_factor(
    basis: &BasisSet,
    atoms: &[Atom],
    baked: Option<SadStampLookup>,
) -> Result<SadFactor> {
    let sites = distinct_stamps(basis, atoms)?;
    let n_ao = basis.nao_sphe() as usize;
    let mut elements = Vec::with_capacity(sites.len());
    // (first AO, element) of each atom that wears a stamp.
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for (e, site) in sites.iter().enumerate() {
        let density = stamp_density(&site.stamp, basis.name(), baked)?;
        let side = density.len().isqrt();
        elements.push(ElementFactor {
            z: site.stamp.z,
            atoms: site.offsets.len(),
            factor: stamp_factor(&density, side),
        });
        placed.extend(site.offsets.iter().map(|&offset| (offset, e)));
    }
    placed.sort_unstable();
    let rank = placed.iter().map(|&(_, e)| elements[e].factor.rank).sum();
    let mut columns = vec![0.0; n_ao * rank];
    let mut column = 0;
    for &(offset, e) in &placed {
        let factor = &elements[e].factor;
        for j in 0..factor.rank {
            let at = offset + n_ao * (column + j);
            columns[at..at + factor.rows]
                .copy_from_slice(&factor.columns[factor.rows * j..factor.rows * (j + 1)]);
        }
        column += factor.rank;
    }
    Ok(SadFactor {
        n_ao,
        rank,
        columns,
        elements,
    })
}

/// `2 L Lᵀ` of the `rows × rank` column-major `columns`, `rows × rows`
/// column-major: what a factor is checked against its density with.
pub fn density_of(columns: &[f64], rows: usize, rank: usize) -> Vec<f64> {
    let mut density = vec![0.0; rows * rows];
    for k in 0..rank {
        let column = &columns[rows * k..rows * (k + 1)];
        for (j, &cj) in column.iter().enumerate() {
            if cj == 0.0 {
                continue;
            }
            for (i, &ci) in column.iter().enumerate() {
                density[i + rows * j] += 2.0 * ci * cj;
            }
        }
    }
    density
}
