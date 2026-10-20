/// @file
/// @brief The H build.

module free_breakfast.integrals.one_electron.core_hamiltonian;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.integrals.one_electron.kinetic;
import free_breakfast.integrals.one_electron.potential_en;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Status core_hamiltonian(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
                        const Molecule &molecule, CartesianMatrixAO &h) {
  // Both checks up front, so a refused input launches nothing: kinetic would
  // otherwise have written T before add_nuclear_attraction refused the nuclei.
  FBF_TRY(check_one_electron_build(res, classes, h, "core_hamiltonian"));
  FBF_TRY(check_nuclei(res, molecule, "core_hamiltonian"));
  // T, then V: each build joins its classes back into res.stream(), so every
  // class's V waits on every class's T.
  FBF_TRY(kinetic(res, classes, h));
  return add_nuclear_attraction(res, classes, molecule, h);
}

} // namespace free_breakfast
