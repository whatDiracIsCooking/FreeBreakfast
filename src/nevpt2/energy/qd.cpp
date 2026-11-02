/// @file
/// @brief The SC-QD-NEVPT2 effective Hamiltonian: its checks over the n x n
/// pair inputs, and the one class walk it shares with class_energies
/// (energy.cpp's sc_couplings).

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

Result<std::vector<ClassEnergies>> sc_effective_hamiltonian(DeviceResources &res,
                                                            const MoBlocks &blocks,
                                                            std::span<const EnergyInputs> pairs,
                                                            const int32_t batch) {
  // The rounded root, checked back: n² pairs, n >= 1.
  const auto n = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(pairs.size()))));
  if (pairs.empty() || std::cmp_not_equal(n * n, pairs.size())) {
    return err_config("sc_effective_hamiltonian: {} pair inputs is not n x n for any n >= 1",
                      pairs.size());
  }
  // The walk reads the shared fields once, from the first pair, so a pair
  // carrying other ones would be silently ignored.
  const EnergyInputs &first = pairs.front();
  const auto shared = [](const EnergyInputs &in) {
    return std::array{in.h1e, in.h2e, in.e_core, in.e_virt, in.h1e_v_sr, in.h1e_v_si, in.h1e_v_sir};
  };
  for (size_t p = 0; p < pairs.size(); ++p) {
    if (shared(pairs[p]) != shared(first)) {
      return err_config("sc_effective_hamiltonian: pair ({}, {}) does not share the first "
                        "pair's integrals and orbital energies",
                        static_cast<int64_t>(p) / n, static_cast<int64_t>(p) % n);
    }
  }
  return sc_couplings(res, blocks, pairs, n, batch);
}

} // namespace free_breakfast::nevpt2
