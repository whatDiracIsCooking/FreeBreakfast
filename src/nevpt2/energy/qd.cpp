/// @file
/// @brief The SC- and PC-QD-NEVPT2 effective Hamiltonians and their reverse
/// passes: their checks over the n x n pair inputs, and the class walks each
/// shares with its one-state energy (energy.cpp's sc_couplings, pc.cpp's
/// pc_couplings, reverse.cpp's sc_couplings_adjoint and
/// pc_couplings_adjoint).

module;

#include "diagnostics.h"

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.nevpt2.density_fitting;
import :shared;

namespace free_breakfast::nevpt2 {

// n, when @p pairs is n x n for some n >= 1 and every pair carries the first
// pair's shared fields; InvalidConfig, naming @p who, otherwise.
Result<int64_t> qd_states(const char *who, std::span<const EnergyInputs> pairs) {
  // The rounded root, checked back: n² pairs, n >= 1.
  const auto n = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(pairs.size()))));
  if (pairs.empty() || std::cmp_not_equal(n * n, pairs.size())) {
    return err_config("{}: {} pair inputs is not n x n for any n >= 1", who, pairs.size());
  }
  // The walk reads the shared fields once, from the first pair, so a pair
  // carrying other ones would be silently ignored.
  const EnergyInputs &first = pairs.front();
  const auto shared = [](const EnergyInputs &in) {
    return std::array{in.h1e, in.h2e, in.e_core, in.e_virt, in.h1e_v_sr, in.h1e_v_si, in.h1e_v_sir};
  };
  for (size_t p = 0; p < pairs.size(); ++p) {
    if (shared(pairs[p]) != shared(first)) {
      return err_config("{}: pair ({}, {}) does not share the first pair's integrals and orbital "
                        "energies",
                        who, static_cast<int64_t>(p) / n, static_cast<int64_t>(p) % n);
    }
  }
  return n;
}

Result<std::vector<ClassEnergies>> sc_effective_hamiltonian(DeviceResources &res,
                                                            const MoBlocks &blocks,
                                                            std::span<const EnergyInputs> pairs,
                                                            const int32_t batch) {
  const int64_t n = FBF_TRY(qd_states("sc_effective_hamiltonian", pairs));
  return sc_couplings(res, blocks, pairs, n, batch);
}

// n, as qd_states, when besides @p hbar is n² and @p bars n long, every entry
// sharing the first's bars on the shared fields; InvalidConfig, naming @p who,
// otherwise. Both reverse passes' checks.
Result<int64_t> qd_adjoint_states(const char *who, std::span<const EnergyInputs> pairs,
                                  std::span<const double> hbar, std::span<const EnergyBars> bars) {
  const int64_t n = FBF_TRY(qd_states(who, pairs));
  if (std::cmp_not_equal(hbar.size(), n * n) || std::cmp_not_equal(bars.size(), n)) {
    return err_config("{}: {} states take {} H-bar elements and {} EnergyBars, not {} and {}", who,
                      n, n * n, n, hbar.size(), bars.size());
  }
  // The walk adds the shared bars once, onto the first entry's.
  const auto shared = [](const EnergyBars &b) {
    return std::array{b.h1e,       b.h2e,    b.h1e_v_sr, b.h1e_v_si,
                      b.h1e_v_sir, b.e_core, b.e_virt,   b.lambda};
  };
  for (size_t j = 0; j < bars.size(); ++j) {
    if (shared(bars[j]) != shared(bars.front())) {
      return err_config("{}: state {}'s bars do not share the first's integral, orbital-energy "
                        "and Λ bars",
                        who, j);
    }
  }
  return n;
}

Result<ScEffectiveHamiltonianAdjoint> sc_effective_hamiltonian_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    std::span<const double> hbar, const int32_t batch, std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition) {
  const int64_t n =
      FBF_TRY(qd_adjoint_states("sc_effective_hamiltonian_adjoint", pairs, hbar, bars));
  return sc_couplings_adjoint(res, blocks, pairs, n, hbar, batch, bars, transition);
}

Result<std::vector<PcClassEnergies>> pc_effective_hamiltonian(DeviceResources &res,
                                                              const MoBlocks &blocks,
                                                              std::span<const EnergyInputs> pairs,
                                                              const int32_t batch) {
  const int64_t n = FBF_TRY(qd_states("pc_effective_hamiltonian", pairs));
  return pc_couplings(res, blocks, pairs, n, batch);
}

Result<PcEffectiveHamiltonianAdjoint> pc_effective_hamiltonian_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    std::span<const double> hbar, const int32_t batch, std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition) {
  const int64_t n =
      FBF_TRY(qd_adjoint_states("pc_effective_hamiltonian_adjoint", pairs, hbar, bars));
  return pc_couplings_adjoint(res, blocks, pairs, n, hbar, batch, bars, transition);
}

} // namespace free_breakfast::nevpt2
