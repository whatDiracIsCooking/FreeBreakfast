// The inverse square root of each element, with a floor: the orthogonalizer's
// map of the eigenvalues to Λ^(-1/2). One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.
//
// Forked from src/recipes/orthogonalize/inverse_sqrt.cu: the element map is
// unchanged, and the launcher is rewritten as fb_inverse_sqrt, inverse_sqrt.h's
// extern "C", enqueue-only launch wrapper, double only (rust/docs/gpu_layer.md,
// "Layering"). The C++ runs the map through wwr's parallel_for, which returns
// the vendor's wwrError_t and consumes the launch error itself; here the same
// index-per-thread launch, four warps a block, is written out, so the status is
// the C ABI's wwr_get_last_error() and no vendor code crosses.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"
#include "primitives.h"

#include "orthogonalize/inverse_sqrt.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <runtime.h>

namespace free_breakfast::device {

namespace {

// data[i] <- data[i] <= floor ? 0 : 1 / sqrt(data[i]), for i below count.
//
// The floor test comes first, so a zero or negative eigenvalue never reaches
// the sqrt. 1 / sqrt rather than rsqrt: the device rsqrt intrinsic is not
// correctly rounded, and this runs once per setup. sqrt is the C library's
// global name, which both vendors declare __host__ __device__ and resolve to
// their correctly rounded device sqrt (primitives.h has the same rule).
//
// Contraction off, under HIP: clang's HIP mode contracts by default
// (-ffp-contract=fast-honor-pragmas), and with the contract flag on both
// operations the AMDGPU back end folds 1 / sqrt(x) into a refined v_rsq_f64
// sequence, which is not the two correctly rounded operations and lands an ulp
// from them on most inputs (gfx1200, clang 23). With it off, the sqrt and the
// division are each correctly rounded, as nvcc's are by default, so both
// backends give the host's 1.0 / x.sqrt() bit for bit. nvcc does no such fold
// and does not know the pragma.
__global__ void inverse_sqrt_kernel(double *const data, const uint32_t count, const double floor) {
#if defined(__HIP__)
#pragma clang fp contract(off)
#endif
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const double value = data[i];
  data[i] = value <= floor ? 0.0 : 1.0 / sqrt(value);
}

} // namespace

} // namespace free_breakfast::device

extern "C" fb_launch_status fb_inverse_sqrt(const wwr_stream stream, const uint32_t count,
                                            double *data, const double floor) {
  using namespace free_breakfast;
  if (count == 0) {
    return FB_LAUNCH_SUCCESS;
  }
  if (data == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The C ABI's stream is the runtime's, behind a void *.
  device::inverse_sqrt_kernel<<<math::idivup(count, kFourWarps), kFourWarps, 0,
                                static_cast<wwr::wwrStream_t>(stream)>>>(data, count, floor);
  // The launch just enqueued, as a status: the C ABI's own wwr_get_last_error(),
  // whose wwr_status values are fb_launch_status's positive range.
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
