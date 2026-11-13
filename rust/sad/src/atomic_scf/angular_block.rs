//! `:angular_block`: one momentum's radial block of the atomic SCF.

use fb_basis::Shell;
use fb_core_error_handling::{Error, Result};

use super::eigen::jacobi_eigen;
use super::gemm::{gemm_nn, gemm_tn};
use super::integrals::AtomicIntegrals;
use crate::aufbau::{AUFBAU_ORDER, subshell_occupation};

/// Linear dependence within one l block, `kLinearDependence`: canonical
/// orthogonalization drops the overlap eigenvectors below this, relative to
/// the largest.
const LINEAR_DEPENDENCE: f64 = 1e-10;

/// One momentum's radial block, `AngularBlock`: the atom's shells of that l,
/// in order, the orthonormal span they cover, and the subshells the aufbau
/// fill seats on its orbitals.
///
/// A spherical density makes the Fock matrix block-diagonal by (l, m) and
/// the same for every m, so each block's matrices are n × n over its shells
/// (the m-average of the full spherical matrix), not `(n (2l+1))²`.
#[derive(Default)]
pub(super) struct AngularBlock {
    /// The momentum.
    pub(super) l: i32,
    /// Each shell's first spherical AO, its `m = −l`.
    pub(super) ao_first: Vec<usize>,
    /// Shells of this l.
    pub(super) n: usize,
    /// n × n.
    pub(super) overlap: Vec<f64>,
    /// n × m canonical orthogonalizer, `Xᵀ S X = 1`.
    pub(super) x: Vec<f64>,
    /// Linearly independent orbitals.
    pub(super) m: usize,
    /// Per orbital, spin-summed, over all 2l+1 m.
    pub(super) occupation: Vec<f64>,
    /// n × n, per m: `Σ_k occ_k / (2l+1) c_k c_kᵀ`.
    pub(super) density: Vec<f64>,
}

impl AngularBlock {
    /// The m-averaged block of the full spherical `nao × nao` matrix:
    /// `average`. Sums over m in the C++ order, then one division.
    pub(super) fn average(&self, full: &[f64], nao: usize) -> Vec<f64> {
        let degen = (2 * self.l + 1) as usize;
        let n = self.n;
        let mut out = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut sum = 0.0;
                for mm in 0..degen {
                    sum += full[(self.ao_first[i] + mm) + (nao * (self.ao_first[j] + mm))];
                }
                out[i + (n * j)] = sum / degen as f64;
            }
        }
        out
    }

    /// Diagonalize `fock` (n × n) in the block's orthonormal span and
    /// refresh [`AngularBlock::density`] from the result: `diagonalize`.
    /// The C++ returns the orbital energies too; nothing here reads them.
    pub(super) fn diagonalize(&mut self, fock: &[f64]) {
        let (n, m) = (self.n, self.m);
        let fx = gemm_nn(fock, &self.x, n, n, m);
        let xfx = gemm_tn(&self.x, &fx, n, m, m);
        let eig = jacobi_eigen(xfx, m);
        let c = gemm_nn(&self.x, &eig.vectors, n, m, m);
        let degen = f64::from(2 * self.l + 1);
        self.density.clear();
        self.density.resize(n * n, 0.0);
        for (k, &occ) in self.occupation.iter().enumerate() {
            let w = occ / degen;
            for j in 0..n {
                for i in 0..n {
                    // c_i c_j grouped before the weight, so the density is
                    // exactly symmetric.
                    self.density[i + (n * j)] += w * (c[i + (n * k)] * c[j + (n * k)]);
                }
            }
        }
    }
}

/// One block per l from 0 to the highest momentum in `shells`, each with
/// its overlap, its orthogonalizer and its share of `z`'s aufbau fill:
/// `angular_blocks`. Densities are left empty, for the caller's first
/// [`AngularBlock::diagonalize`].
///
/// `InvalidConfig` if some l has more occupied subshells than independent
/// orbitals to seat them.
pub(super) fn angular_blocks(
    z: i32,
    shells: &[Shell],
    ints: &AtomicIntegrals,
) -> Result<Vec<AngularBlock>> {
    let lmax = shells.iter().map(Shell::angular_momentum).fold(0, i32::max);
    let mut blocks: Vec<AngularBlock> = (0..=lmax).map(|_| AngularBlock::default()).collect();
    let mut ao = 0;
    for shell in shells {
        blocks[shell.angular_momentum() as usize].ao_first.push(ao);
        ao += shell.function_count_sphe() as usize;
    }

    for (l, block) in (0..).zip(blocks.iter_mut()) {
        block.l = l;
        block.n = block.ao_first.len();
        block.overlap = block.average(&ints.overlap, ints.nao_sphe);
        if block.n == 0 {
            continue;
        }
        let s = jacobi_eigen(block.overlap.clone(), block.n);
        let largest = s.values[block.n - 1];
        for k in 0..block.n {
            if s.values[k] <= LINEAR_DEPENDENCE * largest {
                continue;
            }
            let inv = 1.0 / s.values[k].sqrt();
            for i in 0..block.n {
                block.x.push(s.vectors[i + (block.n * k)] * inv);
            }
            block.m += 1;
        }
    }

    // The k-th subshell of momentum l on the k-th orbital of the l block.
    for (k, &occupation) in subshell_occupation(z).iter().enumerate() {
        let l = AUFBAU_ORDER[k].l as usize;
        if l >= blocks.len() || blocks[l].occupation.len() >= blocks[l].m {
            return Err(Error::config(format!(
                "atomic_scf: Z={z} occupies more l={l} subshells than the basis has independent \
                 orbitals of that momentum"
            )));
        }
        blocks[l].occupation.push(f64::from(occupation));
    }
    Ok(blocks)
}

/// The full spherical `nao × nao` density, each block's
/// [`AngularBlock::density`] on every one of its m: `angular_density`.
pub(super) fn angular_density(blocks: &[AngularBlock], nao: usize) -> Vec<f64> {
    let mut density = vec![0.0; nao * nao];
    for block in blocks {
        let degen = (2 * block.l + 1) as usize;
        for j in 0..block.n {
            for i in 0..block.n {
                for mm in 0..degen {
                    density[(block.ao_first[i] + mm) + (nao * (block.ao_first[j] + mm))] =
                        block.density[i + (block.n * j)];
                }
            }
        }
    }
    density
}
