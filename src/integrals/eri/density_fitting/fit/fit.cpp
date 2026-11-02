/// @file
/// @brief density_fit: the metric's Cholesky factor, then one trsm over (P|mn);
/// or, with lindep set, M^-1/2 over the modes above it, then a gemm per slab.

module free_breakfast.integrals.eri.density_fitting.fit;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // trsm, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.cholesky;
import free_breakfast.recipes.orthogonalize;

#include "diagnostics.h"

namespace free_breakfast {

Status density_fit(DeviceResources &res, const int32_t nao, const int32_t naux, double *metric,
                   double *b, const double lindep) {
  if (nao < 0 || naux < 0) {
    return err_config("density_fit: nao = {} and naux = {} must be non-negative", nao, naux);
  }
  if (!(lindep >= 0.0)) {
    return err_config("density_fit: lindep = {} must be 0 (off) or positive", lindep);
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

  if (lindep > 0.0) {
    // X = M^-1/2 = U Lambda^-1/2 U^T over the modes above lindep, a dropped
    // mode contributing 0: orthogonalize's cut is PySCF's (w > lindep). Into
    // the metric's buffer from a copy of M, freed on the stream after it.
    const size_t nn = static_cast<size_t>(naux) * static_cast<size_t>(naux);
    Buffers::device<double> m(nn, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(m.data(), metric, nn * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(orthogonalize<double>(res, naux, m.data(), metric, lindep));

    // B = X (P|mn), one naux x nao slab (one m) at a time: a gemm into the
    // workspace and a copy back, since a gemm cannot write over its operand.
    const size_t slab = static_cast<size_t>(naux) * static_cast<size_t>(nao);
    const auto [product] = res.device_regions<double>(slab);
    for (int32_t k = 0; k < nao; ++k) {
      double *const b_k = b + (slab * static_cast<size_t>(k));
      FBF_TRY(gpu_status(
          gemm(res.blas(), naux, nao, naux, &kOne<double>, metric, b_k, &kZero<double>, product)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(b_k, product, slab * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
    }
    return {};
  }

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
