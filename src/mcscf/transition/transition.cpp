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
import free_breakfast.core.footprint;
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

MemoryFootprint transition_orbital_gradient_footprint(const int32_t nmo, const int32_t naux,
                                                      const int32_t ncore, const int32_t ncas) {
  const MemoryFootprint fock = response::fock_footprint(nmo, naux, ncore, ncas);
  if (fock == MemoryFootprint{}) {
    return {}; // refused by fock before anything is allocated
  }
  // The transition Fock, held while the zero-RDM one is built; its packed
  // gradient is returned and the rest freed.
  const MemoryFootprint both = then(fock, fock);
  const auto n_mo = static_cast<size_t>(nmo);
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  MemoryFootprint out = both;
  out.device.persistent =
      sizeof(double) * ((n_core * (n_mo - n_core)) + (n_act * (n_mo - n_core - n_act)));
  out.device.scratch = both.device.persistent + both.device.scratch - out.device.persistent;
  return out;
}

} // namespace free_breakfast::mcscf
