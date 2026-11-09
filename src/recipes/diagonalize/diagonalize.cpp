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

// syevd's workspace size for n, in elements of T, queried per call: it is a
// host computation. The legacy API ignores the matrix and eigenvalue pointers
// in a size query.
template <real_fp T>
Result<int32_t> syevd_lwork(DeviceResources &res, const int32_t n) {
  int32_t lwork = 0;
  FBF_TRY(
      gpu_status(wwr::syevd_bufferSize<T>(res.solver().get(), kDiagonalizeEigMode,
                                          kDiagonalizeFillMode, n, nullptr, n, nullptr, &lwork)));
  return lwork;
}

template <real_fp T>
Status diagonalize(DeviceResources &res, const int32_t n, T *a, T *w) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t lwork = FBF_TRY(syevd_lwork<T>(res, n));
  auto work = wwr::extension::reinterpret_buffer_view<T>(
      res.device_workspace(static_cast<size_t>(lwork) * sizeof(T)));
  FBF_TRY(gpu_status(wwr::syevd<T>(res.solver().get(), kDiagonalizeEigMode, kDiagonalizeFillMode, n,
                                   a, n, w, work.data(), lwork, res.info().data())));

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
Result<MemoryFootprint> diagonalize_footprint(DeviceResources &res, const int32_t n) {
  if (n <= 0) {
    return MemoryFootprint{};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t lwork = FBF_TRY(syevd_lwork<T>(res, n));
  MemoryFootprint out;
  out.device.workspace =
      math::align_up(static_cast<size_t>(lwork) * sizeof(T), DeviceResources::kWorkspaceAlignment);
  return out;
}

template Status diagonalize<float>(DeviceResources &, int32_t, float *, float *);
template Status diagonalize<double>(DeviceResources &, int32_t, double *, double *);
template Result<MemoryFootprint> diagonalize_footprint<float>(DeviceResources &, int32_t);
template Result<MemoryFootprint> diagonalize_footprint<double>(DeviceResources &, int32_t);

} // namespace free_breakfast
