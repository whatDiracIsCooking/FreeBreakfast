/// @file
/// @brief density_fit: the metric's Cholesky factor, then one trsm over (P|mn).

module free_breakfast.integrals.eri.density_fitting.fit;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // trsm, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.cholesky;

#include "diagnostics.h"

namespace free_breakfast {

Status density_fit(DeviceResources &res, const int32_t nao, const int32_t naux, double *metric,
                   double *b) {
  if (nao < 0 || naux < 0) {
    return err_config("density_fit: nao = {} and naux = {} must be non-negative", nao, naux);
  }
  // trsm's column count is nao^2; the int32_t dispatch is the one both
  // backends' BLAS carry.
  const int64_t pairs = int64_t{nao} * nao;
  if (pairs > std::numeric_limits<int32_t>::max()) {
    return err_config("density_fit: nao^2 = {} exceeds int32_t", pairs);
  }
  if (nao == 0 || naux == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};

  // M = L L^T in the lower triangle. Not the solver's potrf: on ROCm 7.2 its
  // small-matrix kernel races (recipes/cholesky/README.md, #404).
  const int32_t info = FBF_TRY(cholesky(res, naux, metric));
  if (info > 0) {
    return err_numerical(
        "density_fit: the metric is not positive definite (pivot {} of {}, 1-based)", info, naux);
  }

  // B = L^-1 (P|mn): every column of the naux x nao^2 view at once.
  return gpu_status(wwr::trsm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
      wwr::WWRBLAS_DIAG_NON_UNIT, naux, static_cast<int32_t>(pairs), &kOne<double>, metric, naux, b,
      naux));
}

} // namespace free_breakfast
