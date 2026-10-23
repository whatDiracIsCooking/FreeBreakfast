/// @file
/// @brief diagonalize, and its float / double instantiations.

module free_breakfast.linalg.diagonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// syevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree.
constexpr auto kDiagonalizeEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kDiagonalizeFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;

template <real_fp T>
Status diagonalize(DeviceResources &res, const int32_t n, T *a, T *w) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};

  // syevd's workspace size for n, queried per call: it is a host computation.
  // The legacy API ignores the matrix and eigenvalue pointers in a size query.
  int32_t lwork = 0;
  FBF_TRY(
      gpu_status(wwr::syevd_bufferSize<T>(res.solver().get(), kDiagonalizeEigMode,
                                          kDiagonalizeFillMode, n, nullptr, n, nullptr, &lwork)));
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

template Status diagonalize<float>(DeviceResources &, int32_t, float *, float *);
template Status diagonalize<double>(DeviceResources &, int32_t, double *, double *);

} // namespace free_breakfast
