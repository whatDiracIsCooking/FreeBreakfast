/// @file
/// @brief DensityFittingStep: the aux basis placed, then metric, three_center
/// and density_fit, in that order, on res.stream().

module free_breakfast.pipeline.density_fitting;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemGetInfo, gpu_status
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;

#include "diagnostics.h"

namespace free_breakfast {

Result<DensityFittingOutput>
DensityFittingStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
                        DeviceResources &res) const {
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  // Placed as the orbital basis is: the fitting sets' coefficients are for
  // normalized primitives too.
  const BasisStamp stamp = FBF_TRY(resolve_aux_basis(aux_basis));
  const BasisSet aux =
      FBF_TRY(place(stamp.primitive_normalized(), in.get<MoleculeOutput>().molecule));
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const ScopedDeviceIndex on{res.dev_idx()};

  // B is the largest thing DF holds, and past the device's memory its
  // allocation would abort (CreatePolicy), not return; three_center then
  // counts its own scratch against what B leaves.
  const size_t b_bytes = sizeof(double) * naux * nao * nao;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (b_bytes > free_bytes) {
    return err_unsupported("density_fitting: B over {} orbital and {} fitting functions needs {} "
                           "bytes of device memory, and device {} has {} free",
                           nao, naux, b_bytes, res.dev_idx(), free_bytes);
  }

  MatrixAOPair m(res, aux, FBF_TRY(upload_spherical_transform_matrix(res, aux)));
  FBF_TRY(metric(res, aux, m));
  Buffers::device<double> b(naux * nao * nao, res.device());
  FBF_TRY(three_center(res, basis, in.get<PrimitivePairClassesOutput>().classes, aux, b.data()));
  // L overwrites the metric, which is freed at return, stream-ordered after
  // the trsm that reads it.
  FBF_TRY(density_fit(res, static_cast<int32_t>(nao), static_cast<int32_t>(naux),
                      m.spherical().data(), b.data()));
  return DensityFittingOutput{.b = std::move(b), .naux = static_cast<int32_t>(naux)};
}

} // namespace free_breakfast
