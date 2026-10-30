/// @file
/// @brief lu_solve: getrf, its flag read, then getrs.

module free_breakfast.recipes.lu;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // getrf, getrs, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

Result<int32_t> lu_solve(DeviceResources &res, const int32_t n, const int32_t nrhs, double *a,
                         double *b) {
  if (n < 0 || nrhs < 0) {
    return err_config("lu_solve: needs n >= 0 and nrhs >= 0 (n = {}, nrhs = {})", n, nrhs);
  }
  if (n == 0) {
    return 0;
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  wwr::wwrsolverDnHandle_t solver = res.solver().get();

  // getrf's work for n, queried per call: a host computation.
  int32_t lwork = 0;
  FBF_TRY(gpu_status(wwr::getrf_bufferSize<double>(solver, n, n, a, n, &lwork)));
  const auto [work, pivots] =
      res.device_regions<double, int32_t>(static_cast<size_t>(lwork), static_cast<size_t>(n));
  FBF_TRY(gpu_status(wwr::getrf<double>(solver, n, n, a, n, work, pivots, res.info().data())));
  const int32_t factored = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug here, not in A); info > 0 is an
  // exactly zero pivot, which is A's doing and the caller's to act on.
  if (factored < 0) {
    return err_gpu("lu_solve: getrf rejected argument {}", -factored);
  }
  if (factored > 0 || nrhs == 0) {
    return factored;
  }
  FBF_TRY(gpu_status(wwr::getrs<double>(solver, wwr::WWRBLAS_OP_N, n, nrhs, a, n, pivots, b, n,
                                        res.info().data())));
  if (const int32_t solved = FBF_TRY(res.fetch_info()); solved != 0) {
    return err_gpu("lu_solve: getrs rejected argument {}", -solved);
  }
  return 0;
}

} // namespace free_breakfast
