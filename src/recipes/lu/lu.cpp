/// @file
/// @brief lu_solve: getrf, its flag read, then the solve of our own: lu.cu's
/// row interchanges and two trsm recipe calls.

module;

// The launcher, defined in lu.cu: declared in the GMF so it keeps external
// linkage (see the header).
#include "lu_bridge.h"

module free_breakfast.recipes.lu;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // getrf, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.recipes.trsm;

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
  const size_t lwork = FBF_TRY(getrf_lwork(res, n));
  const auto [work, pivots] =
      FBF_TRY((res.device_regions<double, int32_t>("lu_solve", lwork, static_cast<size_t>(n))));
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
  // getrs, by hand: B <- P B, then L^-1 B (unit diagonal), then U^-1 B. The
  // library's getrs is not used because hipSOLVER's allocates past its size
  // query and keeps it (the README).
  FBF_TRY(gpu_status(device::lu_row_interchanges(res.stream(), n, nrhs, pivots, b, n)));
  FBF_TRY(trsm<double>(res, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
                       wwr::WWRBLAS_DIAG_UNIT, n, nrhs, 1.0, a, n, b, n));
  FBF_TRY(trsm<double>(res, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_UPPER, wwr::WWRBLAS_OP_N,
                       wwr::WWRBLAS_DIAG_NON_UNIT, n, nrhs, 1.0, a, n, b, n));
  return 0;
}

Result<size_t> getrf_lwork(DeviceResources &res, const int32_t n) {
  if (n <= 0) {
    return size_t{0};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  int32_t lwork = 0;
  FBF_TRY(gpu_status(wwr::getrf_bufferSize<double>(res.solver().get(), n, n, nullptr, n, &lwork)));
  return static_cast<size_t>(lwork);
}

MemoryFootprint lu_solve_footprint(const int32_t n, const size_t lwork) {
  if (n <= 0) {
    return {};
  }
  // device_regions' carve: each region rounded up to the alignment. The solve
  // after getrf adds nothing: its kernel takes no scratch, and the trsm
  // recipe only the gemm the BLAS handle's fixed workspace serves.
  constexpr size_t kAlign = DeviceResources::kWorkspaceAlignment;
  MemoryFootprint out;
  out.device.workspace = math::align_up(lwork * sizeof(double), kAlign) +
                         math::align_up(static_cast<size_t>(n) * sizeof(int32_t), kAlign);
  return out;
}

} // namespace free_breakfast
