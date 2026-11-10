/// @file
/// @brief diagonalize, and its float / double instantiations.

module free_breakfast.recipes.diagonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives;
import free_breakfast.core.footprint;

#include "diagnostics.h"

namespace free_breakfast {

// syevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree.
constexpr auto kDiagonalizeEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kDiagonalizeFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;

// The legacy API ignores the matrix and eigenvalue pointers in a size query.
template <real_fp T>
Result<size_t> syevd_lwork(DeviceResources &res, const int32_t n) {
  if (n <= 0) {
    return size_t{0};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  int32_t lwork = 0;
  FBF_TRY(
      gpu_status(wwr::syevd_bufferSize<T>(res.solver().get(), kDiagonalizeEigMode,
                                          kDiagonalizeFillMode, n, nullptr, n, nullptr, &lwork)));
  return static_cast<size_t>(lwork);
}

template <real_fp T>
Status diagonalize(DeviceResources &res, const int32_t n, T *a, T *w) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t lwork = FBF_TRY(syevd_lwork<T>(res, n));
  const auto [work] = FBF_TRY(res.device_regions<T>("diagonalize", lwork));
  FBF_TRY(gpu_status(wwr::syevd<T>(res.solver().get(), kDiagonalizeEigMode, kDiagonalizeFillMode, n,
                                   a, n, w, work, static_cast<int32_t>(lwork), res.info().data())));

  const int32_t info = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug in the caller, not in A); info > 0 is
  // non-convergence, which is A's doing.
  if (info < 0) {
    return err_gpu("diagonalize: syevd rejected argument {}", -info);
  }
  if (info > 0) {
    return err_numerical("diagonalize: syevd did not converge (info = {})", info);
  }
  return {};
}

template <real_fp T>
MemoryFootprint diagonalize_footprint(const size_t lwork) {
  MemoryFootprint out;
  out.device.workspace = math::align_up(lwork * sizeof(T), DeviceResources::kWorkspaceAlignment);
  return out;
}

template Status diagonalize<float>(DeviceResources &, int32_t, float *, float *);
template Status diagonalize<double>(DeviceResources &, int32_t, double *, double *);
template Result<size_t> syevd_lwork<float>(DeviceResources &, int32_t);
template Result<size_t> syevd_lwork<double>(DeviceResources &, int32_t);
template MemoryFootprint diagonalize_footprint<float>(size_t);
template MemoryFootprint diagonalize_footprint<double>(size_t);

} // namespace free_breakfast
