/// @file
/// @brief atomic_scf: the Fock build, the energy, the per-l commutators, DIIS.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :integrals;
import :gemm;
import :diis;
import :angular_block;
import :scf;

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

Result<AtomicScf> atomic_scf(int32_t z, std::span<const Shell> shells, double tolerance,
                             int32_t max_iterations) {
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
