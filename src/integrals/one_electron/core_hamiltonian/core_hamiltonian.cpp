/// @file
/// @brief The H build.

module free_breakfast.integrals.one_electron.core_hamiltonian;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.integrals.one_electron.kinetic;
import free_breakfast.integrals.one_electron.potential_en;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Status core_hamiltonian(DeviceResources &res,
                        const std::span<const DevicePrimitivePairClass> classes,
                        const DeviceNuclei &nuclei, CartesianMatrixAO &h) {
  // Both checks up front, so a refused input launches nothing: kinetic would
  // otherwise have written T before add_nuclear_attraction refused the nuclei.
  FBF_TRY(check_one_electron_build(res, classes, h, "core_hamiltonian"));
  FBF_TRY(check_nuclei(res, nuclei, "core_hamiltonian"));
  FBF_TRY(kinetic(res, classes, h));
  return add_nuclear_attraction(res, classes, nuclei, h);
}

} // namespace free_breakfast
