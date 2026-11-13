//! `:scf` and the partitions under `atomic_scf/`: one atom's SCF,
//! [`AtomicScf`] and [`atomic_scf`].

mod angular_block;
mod diis;
mod eigen;
mod eri;
mod gemm;
mod hermite;
mod integrals;
mod one_electron;

use fb_basis::Shell;
use fb_core_error_handling::{Error, Result};

use angular_block::{angular_blocks, angular_density};
use diis::AtomicDiis;
use gemm::gemm_nn;
use integrals::atomic_integrals;

/// The DIIS history length, `kDiisDepth`.
const DIIS_DEPTH: usize = 8;

/// The C++ defaults of `atomic_scf`'s last two parameters: the tolerance on
/// the commutator and the energy change (hartree), and the Fock builds
/// before giving up.
const TOLERANCE: f64 = 1e-9;
const MAX_ITERATIONS: i32 = 128;

/// `F = H + J − K/2` over the spin-summed density, `J_ij = (ij|kl) D_kl`,
/// `K_ij = (ik|jl) D_kl`: `rhf_fock`. Both sums run in the C++ order.
fn rhf_fock(hcore: &[f64], eri: &[f64], density: &[f64], nao: usize) -> Vec<f64> {
    let mut fock = hcore.to_vec();
    for j in 0..nao {
        for i in 0..nao {
            let mut coulomb = 0.0;
            let mut exchange = 0.0;
            for l in 0..nao {
                for k in 0..nao {
                    let d = density[k + (nao * l)];
                    coulomb += eri[i + (nao * (j + (nao * (k + (nao * l)))))] * d;
                    exchange += eri[i + (nao * (k + (nao * (j + (nao * l)))))] * d;
                }
            }
            fock[i + (nao * j)] += coulomb - (0.5 * exchange);
        }
    }
    fock
}

/// A converged atomic SCF: `AtomicScf`, without the orbital energies no
/// caller reads.
pub(crate) struct AtomicScf {
    /// Total electronic energy, hartree (an atom has no nuclear repulsion).
    pub(crate) energy: f64,
    /// Fock builds it took.
    pub(crate) iterations: i32,
    /// Spin-summed density over the atom's spherical layout, n × n,
    /// column-major: its trace against the atom's overlap is z.
    pub(crate) density: Vec<f64>,
}

/// The spherically averaged, fractionally occupied RHF of a neutral atom of
/// atomic number `z` in `shells`, at the C++ defaults (1e-9, 128 builds):
/// `atomic_scf`.
pub(crate) fn atomic_scf(z: i32, shells: &[Shell]) -> Result<AtomicScf> {
    atomic_scf_to(z, shells, TOLERANCE, MAX_ITERATIONS)
}

/// [`atomic_scf`] at a given `tolerance` and `max_iterations`.
///
/// Starts from the core Hamiltonian, accelerates with Pulay DIIS on each l
/// block's commutator `F P S − S P F`, and stops when the largest commutator
/// element and the energy change both fall under `tolerance`. `z` is in
/// [0, 118] (outside aborts); 0 gives an empty density.
///
/// `InvalidConfig` as `atomic_integrals`, or if some l has more occupied
/// subshells than linearly independent orbitals to seat them (an SCF over
/// the rest would be of a different ion); `Unsupported` as
/// `atomic_integrals`; `MaxIterationsExceeded` if it has not converged by
/// then.
fn atomic_scf_to(
    z: i32,
    shells: &[Shell],
    tolerance: f64,
    max_iterations: i32,
) -> Result<AtomicScf> {
    let ints = atomic_integrals(z, shells)?;
    let nao = ints.nao_sphe;
    let mut blocks = angular_blocks(z, shells, &ints)?;

    let hcore: Vec<f64> = ints
        .kinetic
        .iter()
        .zip(&ints.nuclear)
        .map(|(t, v)| t + v)
        .collect();
    for block in &mut blocks {
        if block.m > 0 {
            let h = block.average(&hcore, nao);
            block.diagonalize(&h);
        }
    }

    let mut diis = AtomicDiis::new(DIIS_DEPTH);
    let mut previous_energy = 0.0;
    for iteration in 1..=max_iterations {
        let density = angular_density(&blocks, nao);
        let fock = rhf_fock(&hcore, &ints.eri, &density, nao);
        let mut energy = 0.0;
        for i in 0..nao * nao {
            energy += 0.5 * density[i] * (hcore[i] + fock[i]);
        }

        // Per block: the averaged Fock matrix and its commutator with P
        // against S.
        let mut stacked_fock = Vec::new();
        let mut stacked_error = Vec::new();
        let mut max_error: f64 = 0.0;
        for block in &blocks {
            let n = block.n;
            let f = block.average(&fock, nao);
            let fp = gemm_nn(&f, &block.density, n, n, n);
            let fps = gemm_nn(&fp, &block.overlap, n, n, n);
            for j in 0..n {
                for i in 0..n {
                    // S P F = (F P S)ᵀ: every factor is symmetric.
                    let e = fps[i + (n * j)] - fps[j + (n * i)];
                    stacked_error.push(e);
                    max_error = max_error.max(e.abs());
                }
            }
            stacked_fock.extend_from_slice(&f);
        }

        if max_error < tolerance && (energy - previous_energy).abs() < tolerance {
            return Ok(AtomicScf {
                energy,
                iterations: iteration,
                density,
            });
        }
        previous_energy = energy;

        let extrapolated = diis.extrapolate(stacked_fock, stacked_error);
        let mut at = 0;
        for block in &mut blocks {
            let first = at;
            at += block.n * block.n;
            if block.m > 0 {
                block.diagonalize(&extrapolated[first..at]);
            }
        }
    }
    Err(Error::max_iterations_exceeded(format!(
        "atomic_scf: Z={z} not converged to {tolerance} in {max_iterations} iterations"
    )))
}

/// The refusals `sad_density` cannot reach through a placed basis, and the
/// SCF's own: unit tests, so the SCF stays private.
#[cfg(test)]
mod tests {
    use fb_basis::{Primitive, Shell};
    use fb_core_error_handling::ErrorKind;
    use fb_core_types::Double3;

    use super::*;

    fn s_shell(exponent: f64, center: Double3) -> Shell {
        Shell::new(
            0,
            vec![Primitive {
                exponent,
                coefficient: 1.0,
            }],
            center,
        )
    }

    /// `atomic_integrals` refuses no shells: `sad_density` skips an atom
    /// with none, so only a direct call reaches it.
    #[test]
    fn refuses_no_shells() {
        assert_eq!(
            atomic_scf(1, &[]).err().map(|e| e.kind),
            Some(ErrorKind::InvalidConfig)
        );
    }

    /// Shells on two centers are not one atom's.
    #[test]
    fn refuses_shells_on_two_centers() {
        let shells = [
            s_shell(1.0, Double3::default()),
            s_shell(
                1.0,
                Double3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ),
        ];
        assert_eq!(
            atomic_scf(1, &shells).err().map(|e| e.kind),
            Some(ErrorKind::InvalidConfig)
        );
    }

    /// An SCF given one Fock build cannot have converged.
    #[test]
    fn refuses_an_scf_out_of_iterations() {
        let shells = [
            s_shell(5.0, Double3::default()),
            s_shell(0.5, Double3::default()),
        ];
        let error = atomic_scf_to(2, &shells, TOLERANCE, 1).err().unwrap();
        assert_eq!(error.kind, ErrorKind::MaxIterationsExceeded);
        assert!(atomic_scf(2, &shells).is_ok());
    }
}
