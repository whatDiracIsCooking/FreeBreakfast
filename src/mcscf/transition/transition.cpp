/// @file
/// @brief mcscf::transition_orbital_gradient: two Focks' gradients'
/// difference, as df_nac formed it before it was hoisted here.

module free_breakfast.mcscf.transition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.response.fock;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

Result<Buffers::device<double>> transition_orbital_gradient(
    DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const int32_t naux, const SphericalMatrixAO &c,
    const int32_t ncore, const int32_t ncas, const Buffers::device<double> &ppaa,
    const Buffers::device<double> &papa, const std::span<const double> gamma,
    const std::span<const double> gamma2) {
  response::Fock transition =
      FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas, ppaa, papa, gamma, gamma2));
  // fock checks ncas against nmo, so the zero RDMs' sizes are its own.
  const auto m = static_cast<size_t>(ncas) * static_cast<size_t>(ncas);
  const std::vector<double> zero_gamma(m, 0.0);
  const std::vector<double> zero_gamma2(m * m, 0.0);
  const response::Fock core = FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas,
                                                     ppaa, papa, zero_gamma, zero_gamma2));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(
      res.blas().get(), static_cast<int32_t>(transition.gradient.num_elements()), &kNegOne<double>,
      core.gradient.data(), 1, transition.gradient.data(), 1)));
  // Waited for: the axpy reads core's gradient, freed on return.
  FBF_TRY(res.sync());
  return std::move(transition.gradient);
}

} // namespace free_breakfast::mcscf
