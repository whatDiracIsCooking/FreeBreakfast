// The device half of free_breakfast.nevpt2.density_fitting: the permute that
// takes a slab's gemm product into the block's physicist layout. One source
// for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "nevpt2_permute_bridge.h"

namespace free_breakfast::nevpt2::device {

namespace {

// One thread per output element, reading the input at its strides: the writes
// coalesce, the reads gather. A slab is read once, so no tiling.
__global__ void nevpt2_permute_kernel(const int64_t n0, const int64_t n1, const int64_t n2,
                                      const int64_t count, const int64_t s0, const int64_t s1,
                                      const int64_t s2, const int64_t s3, const double *const in,
                                      double *const out) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  int64_t rest = i;
  const int64_t o0 = rest % n0;
  rest /= n0;
  const int64_t o1 = rest % n1;
  rest /= n1;
  const int64_t o2 = rest % n2;
  const int64_t o3 = rest / n2;
  out[i] = in[(o0 * s0) + (o1 * s1) + (o2 * s2) + (o3 * s3)];
}

} // namespace

wwr::wwrError_t permute(const wwr::wwrStream_t stream, const int64_t n0, const int64_t n1,
                        const int64_t n2, const int64_t n3, const int64_t s0, const int64_t s1,
                        const int64_t s2, const int64_t s3, const double *const in,
                        double *const out) {
  const int64_t count = n0 * n1 * n2 * n3;
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  nevpt2_permute_kernel<<<static_cast<uint32_t>(math::idivup(count, int64_t{kFourWarps})),
                          kFourWarps, 0, stream>>>(n0, n1, n2, count, s0, s1, s2, s3, in, out);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::nevpt2::device
