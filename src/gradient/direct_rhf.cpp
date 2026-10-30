/// @file
/// @brief rhf: the shared one-electron terms, then the integral-direct J and K
/// gradients on D, K's with RHF's -1/2.

module free_breakfast.gradient;

import std;
import free_breakfast.core.constants; // kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::gradient {

Result<RhfTerms> rhf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                     const std::span<const PrimitivePairClass> classes,
                     const std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
                     const std::span<const double> mo_energies, const int32_t nocc) {
  RhfTerms g;
  const SphericalMatrixAO d =
      FBF_TRY(one_electron_terms(res, molecule, basis, classes, c, mo_energies, nocc, g));
  g.coulomb = FBF_TRY(coulomb_gradient(res, basis, classes, d));
  g.exchange = FBF_TRY(exchange_gradient(res, basis, k_classes, d));
  for (double &x : g.exchange) {
    x *= kNegHalf<double>;
  }

  g.total = g.nuclear_repulsion;
  for (const std::vector<double> *term :
       {&g.kinetic, &g.potential, &g.overlap, &g.coulomb, &g.exchange}) {
    g.total = sum_of(std::move(g.total), *term);
  }
  return g;
}

} // namespace free_breakfast::gradient
