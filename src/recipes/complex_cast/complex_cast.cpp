/// @file
/// @brief split_planes and merge_planes, and their wwrFloatComplex /
/// wwrDoubleComplex instantiations.

module;

// The kernel launchers, defined in complex_cast.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "complex_cast_bridge.h"

module free_breakfast.recipes.complex_cast;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

namespace free_breakfast {

template <wwr::complex_fp Z>
Status split_planes(DeviceResources &res, const int32_t rows, const int32_t cols, const Z *a,
                    const int32_t lda, wwr::ComplexToRealType<Z> *re, wwr::ComplexToRealType<Z> *im,
                    const int32_t ldp) {
  if (rows <= 0 || cols <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(static_cast<wwr::wwrError_t>(
      device::split_planes(res.stream(), rows, cols, a, lda, re, im, ldp)));
}

template <wwr::complex_fp Z>
Status merge_planes(DeviceResources &res, const int32_t rows, const int32_t cols,
                    const wwr::ComplexToRealType<Z> *re, const wwr::ComplexToRealType<Z> *im,
                    const int32_t ldp, Z *c, const int32_t ldc) {
  if (rows <= 0 || cols <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(static_cast<wwr::wwrError_t>(
      device::merge_planes(res.stream(), rows, cols, re, im, ldp, c, ldc)));
}

template Status split_planes<wwr::wwrFloatComplex>(DeviceResources &, int32_t, int32_t,
                                                   const wwr::wwrFloatComplex *, int32_t, float *,
                                                   float *, int32_t);
template Status split_planes<wwr::wwrDoubleComplex>(DeviceResources &, int32_t, int32_t,
                                                    const wwr::wwrDoubleComplex *, int32_t,
                                                    double *, double *, int32_t);
template Status merge_planes<wwr::wwrFloatComplex>(DeviceResources &, int32_t, int32_t,
                                                   const float *, const float *, int32_t,
                                                   wwr::wwrFloatComplex *, int32_t);
template Status merge_planes<wwr::wwrDoubleComplex>(DeviceResources &, int32_t, int32_t,
                                                    const double *, const double *, int32_t,
                                                    wwr::wwrDoubleComplex *, int32_t);

} // namespace free_breakfast
