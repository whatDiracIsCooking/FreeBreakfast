/// @file
/// @brief Both symmetrize overloads: the argument checks, then one launch of
/// symmetrize.cu's kernels.

module;

// The launchers, defined in symmetrize.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "symmetrize_bridge.h"

module free_breakfast.recipes.symmetrize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrblasFillMode_t, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;

namespace free_breakfast {

Status symmetrize(DeviceResources &res, const int32_t n, double *a, const int32_t lda) {
  if (n < 0 || lda < n) {
    return err_config("symmetrize: n = {} must be non-negative and lda = {} at least n", n, lda);
  }
  if (n == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::symmetrize(res.stream(), n, a, lda));
}

Status symmetrize(DeviceResources &res, const wwr::wwrblasFillMode_t fill, const int32_t n,
                  double *a, const int32_t lda) {
  if (fill != wwr::WWRBLAS_FILL_MODE_LOWER && fill != wwr::WWRBLAS_FILL_MODE_UPPER) {
    return err_config("symmetrize: fill must be WWRBLAS_FILL_MODE_LOWER or _UPPER");
  }
  if (n < 0 || lda < n) {
    return err_config("symmetrize: n = {} must be non-negative and lda = {} at least n", n, lda);
  }
  if (n == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(
      device::mirror_triangle(res.stream(), fill == wwr::WWRBLAS_FILL_MODE_LOWER, n, a, lda));
}

} // namespace free_breakfast
