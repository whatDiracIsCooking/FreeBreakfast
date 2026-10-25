// The device half of free_breakfast.mcscf.solve: the augmented Hessian's
// diagonal preconditioner, so its Davidson's corrections never leave the
// device. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "ah_precondition_bridge.h"

namespace free_breakfast::device {

namespace {

__global__ void ah_precondition_kernel(const int32_t n, const double theta, const double floor,
                                       const double *const diagonal, const double *const residual,
                                       double *const correction) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n) {
    return;
  }
  double d = theta - diagonal[i];
  if (d < floor && d > -floor) {
    d = d < 0.0 ? -floor : floor;
  }
  correction[i] = residual[i] * (1.0 / d);
}

} // namespace

wwr::wwrError_t ah_precondition(const wwr::wwrStream_t stream, const int32_t n, const double theta,
                                const double floor, const double *const diagonal,
                                const double *const residual, double *const correction) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  ah_precondition_kernel<<<static_cast<uint32_t>(math::idivup(int64_t{n}, int64_t{kFourWarps})),
                           kFourWarps, 0, stream>>>(n, theta, floor, diagonal, residual,
                                                    correction);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
