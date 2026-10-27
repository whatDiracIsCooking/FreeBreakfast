// The device half of free_breakfast.recipes.norm: the abs_at kernel. One
// source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "primitives.h"

#include "abs_at_bridge.h"

namespace free_breakfast::device {

namespace {

// result[0] <- |x[(*at - 1) * incx]|. One element, so one thread of one block.
// The offset is 64-bit, so a large strided index cannot overflow.
template <typename T>
__global__ void abs_at_kernel(const T *const x, const int64_t incx, const int32_t *const at,
                              T *const result) {
  result[0] = math::fabs(x[(static_cast<int64_t>(*at) - 1) * incx]);
}

} // namespace

template <typename T>
wwr::wwrError_t abs_at(const wwr::wwrStream_t stream, const T *x, const int32_t incx,
                       const int32_t *at, T *result) {
  abs_at_kernel<<<1, 1, 0, stream>>>(x, incx, at, result);
  return wwr::wwrGetLastError();
}

template wwr::wwrError_t abs_at<float>(wwr::wwrStream_t, const float *, int32_t, const int32_t *,
                                       float *);
template wwr::wwrError_t abs_at<double>(wwr::wwrStream_t, const double *, int32_t, const int32_t *,
                                        double *);

} // namespace free_breakfast::device
