/// @file
/// @brief One atom's SCF: @ref free_breakfast::AtomicScf and @ref
/// free_breakfast::atomic_scf.
module free_breakfast.sad:scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :integrals;
import :gemm;
import :diis;
import :angular_block;

#include "diagnostics.h"

namespace free_breakfast {

// DIIS history length.
constexpr size_t kDiisDepth = 8;

// F = H + J - K/2 over the spin-summed density, J_ij = (ij|kl) D_kl,
// K_ij = (ik|jl) D_kl.
std::vector<double> rhf_fock(const std::vector<double> &hcore, const std::vector<double> &eri,
                             const std::vector<double> &density, size_t nao) {
  std::vector<double> fock = hcore;
  for (size_t j = 0; j < nao; ++j) {
    for (size_t i = 0; i < nao; ++i) {
      double coulomb = 0.0;
      double exchange = 0.0;
      for (size_t l = 0; l < nao; ++l) {
        for (size_t k = 0; k < nao; ++k) {
          const double d = density[k + (nao * l)];
          coulomb += eri[i + (nao * (j + (nao * (k + (nao * l)))))] * d;
          exchange += eri[i + (nao * (k + (nao * (j + (nao * l)))))] * d;
        }
      }
      fock[i + (nao * j)] += coulomb - (0.5 * exchange);
    }
  }
  return fock;
}

/// @brief A converged atomic SCF.
struct AtomicScf {
  double energy{0.0};    ///< Total electronic energy, hartree (an atom has no nuclear repulsion).
  int32_t iterations{0}; ///< Fock builds it took.
  /// Spin-summed density over the atom's spherical layout, n x n,
  /// column-major: its trace against @ref AtomicIntegrals::overlap is z.
  std::vector<double> density;
  /// Per l, from 0 to the highest momentum of the shells, the l block's orbital
  /// energies ascending (one per m-degenerate level; empty for a momentum with
  /// no shell).
  std::vector<std::vector<double>> orbital_energies;
};

/// @brief The spherically averaged, fractionally occupied RHF of a neutral atom
/// of atomic number @p z in @p shells.
///
/// Starts from the core Hamiltonian, accelerates with Pulay DIIS on each l
/// block's commutator F P S - S P F, and stops when the largest commutator
/// element and the energy change both fall under @p tolerance.
///
/// @param z              Atomic number in [0, 118] (outside aborts, as for
///                       @ref subshell_occupation); 0 gives an empty density.
/// @param shells         The atom's shells, as for @ref atomic_integrals.
/// @param tolerance      On the commutator and the energy change, hartree.
/// @param max_iterations Fock builds before giving up.
/// @return ErrorKind::InvalidConfig, as @ref atomic_integrals, or if some l
///   has more occupied subshells than linearly independent orbitals to seat
///   them (an SCF over the rest would be of a different ion).
///   ErrorKind::Unsupported as @ref atomic_integrals.
///   ErrorKind::MaxIterationsExceeded if it has not converged by then.
[[nodiscard]] Result<AtomicScf> atomic_scf(int32_t z, std::span<const Shell> shells,
                                           double tolerance = 1e-9, int32_t max_iterations = 128) {
  const AtomicIntegrals ints = FBF_TRY(atomic_integrals(z, shells));
  const size_t nao = ints.nao_sphe;
  std::vector<AngularBlock> blocks = FBF_TRY(angular_blocks(z, shells, ints));

  std::vector<double> hcore(nao * nao);
  for (size_t i = 0; i < hcore.size(); ++i) {
    hcore[i] = ints.kinetic[i] + ints.nuclear[i];
  }
  for (AngularBlock &block : blocks) {
    if (block.m > 0) {
      (void)block.diagonalize(block.average(hcore, nao));
    }
  }

  AtomicDiis diis(kDiisDepth);
  double previous_energy = 0.0;
  for (int32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    std::vector<double> density = angular_density(blocks, nao);
    const std::vector<double> fock = rhf_fock(hcore, ints.eri, density, nao);
    double energy = 0.0;
    for (size_t i = 0; i < nao * nao; ++i) {
      energy += 0.5 * density[i] * (hcore[i] + fock[i]);
    }

    // Per block: the averaged Fock matrix and its commutator with P against S.
    std::vector<double> stacked_fock;
    std::vector<double> stacked_error;
    double max_error = 0.0;
    for (const AngularBlock &block : blocks) {
      const std::vector<double> f = block.average(fock, nao);
      const std::vector<double> fp = gemm_nn(f, block.density, block.n, block.n, block.n);
      const std::vector<double> fps = gemm_nn(fp, block.overlap, block.n, block.n, block.n);
      for (size_t j = 0; j < block.n; ++j) {
        for (size_t i = 0; i < block.n; ++i) {
          // S P F = (F P S)^T: every factor is symmetric.
          const double e = fps[i + (block.n * j)] - fps[j + (block.n * i)];
          stacked_error.push_back(e);
          max_error = std::max(max_error, std::abs(e));
        }
      }
      stacked_fock.insert(stacked_fock.end(), f.begin(), f.end());
    }

    if (max_error < tolerance && std::abs(energy - previous_energy) < tolerance) {
      AtomicScf out{.energy = energy, .iterations = iteration, .density = std::move(density)};
      for (AngularBlock &block : blocks) {
        out.orbital_energies.push_back(block.m > 0 ? block.diagonalize(block.average(fock, nao))
                                                   : std::vector<double>{});
      }
      return out;
    }
    previous_energy = energy;

    const std::vector<double> extrapolated =
        diis.extrapolate(std::move(stacked_fock), std::move(stacked_error));
    size_t at = 0;
    for (AngularBlock &block : blocks) {
      const auto first = extrapolated.begin() + static_cast<std::ptrdiff_t>(at);
      at += block.n * block.n;
      if (block.m > 0) {
        (void)block.diagonalize(
            std::vector<double>(first, extrapolated.begin() + static_cast<std::ptrdiff_t>(at)));
      }
    }
  }
  return err_max_iterations_exceeded("atomic_scf: Z={} not converged to {} in {} iterations", z,
                                     tolerance, max_iterations);
}

} // namespace free_breakfast
