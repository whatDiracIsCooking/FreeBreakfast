/// @file
/// @brief one_electron_terms: D and -W from the occupied orbitals (a gemm and
/// a gmmwich), then, in the exported overload at any D and -W, S, T and V's
/// derivative contractions and the nuclear repulsion's.

module free_breakfast.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::gradient {

Result<SphericalMatrixAO>
one_electron_terms(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &c,
                   const std::span<const double> mo_energies, const int32_t nocc, RhfTerms &g) {
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("gradient: C is on device {}, but the call passed device {}", c.dev_idx(),
                      res.dev_idx());
  }
  const uint32_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("gradient: C is {} square, but the basis has {} spherical AOs", c.n(), nao);
  }
  if (mo_energies.size() != nao) {
    return err_config("gradient: {} orbital energies for {} orbitals", mo_energies.size(), nao);
  }
  if (nocc < 0 || static_cast<uint32_t>(nocc) > nao) {
    return err_config("gradient: {} occupied orbitals of {}", nocc, nao);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  SphericalMatrixAO d(res, basis);
  SphericalMatrixAO minus_w(res, basis);
  // -W = C_occ diag(-2 eps_occ) C_occ^T, through gmmwich's nao x nocc work.
  // The host diagonal is read by its copy before overlap_gradient waits on the
  // stream, which is also before the device buffers are freed.
  std::vector<double> diagonal(static_cast<size_t>(nocc));
  std::ranges::transform(mo_energies.first(diagonal.size()), diagonal.begin(),
                         [](const double e) { return -2.0 * e; });
  Buffers::device<double> d_diagonal(diagonal.size(), res.device());
  Buffers::device<double> work(static_cast<size_t>(nao) * diagonal.size(), res.device());
  if (nocc > 0) {
    const auto n = static_cast<int32_t>(nao);
    // D = 2 C_occ C_occ^T.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), n, n, nocc, &kTwo<double>, c.data(), c.data(), &kZero<double>, d.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_diagonal.data(), diagonal.data(),
                                           sizeof(double) * diagonal.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(gpu_status(gmmwich<wwr::WWRBLAS_OP_T>(res.blas(), n, nocc, c.data(), d_diagonal.data(),
                                                  work.data(), minus_w.data())));
  }
  FBF_TRY(one_electron_terms(res, molecule, basis, classes, d, minus_w, g));
  return d;
}

Status one_electron_terms(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                          const std::span<const PrimitivePairClass> classes,
                          const SphericalMatrixAO &d, const SphericalMatrixAO &minus_w,
                          RhfTerms &g) {
  const ScopedDeviceIndex on{res.dev_idx()};
  g.overlap = FBF_TRY(overlap_gradient(res, basis, classes, minus_w));
  g.kinetic = FBF_TRY(kinetic_gradient(res, basis, classes, d));
  g.potential = FBF_TRY(potential_gradient(res, basis, classes, molecule, d));
  g.nuclear_repulsion = nuclear_repulsion_gradient(molecule);
  return {};
}

} // namespace free_breakfast::gradient
