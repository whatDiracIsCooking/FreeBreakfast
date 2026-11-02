/// @file
/// @brief The SC- and PC-QD-NEVPT2 effective Hamiltonians: their checks over
/// the n x n pair inputs, and the class walks each shares with its one-state
/// energy (energy.cpp's sc_couplings, pc.cpp's pc_couplings).

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

Result<std::vector<PcClassEnergies>> pc_effective_hamiltonian(DeviceResources &res,
                                                              const MoBlocks &blocks,
                                                              std::span<const EnergyInputs> pairs,
                                                              const int32_t batch) {
  const int64_t n = FBF_TRY(qd_states("pc_effective_hamiltonian", pairs));
  return pc_couplings(res, blocks, pairs, n, batch);
}

} // namespace free_breakfast::nevpt2
