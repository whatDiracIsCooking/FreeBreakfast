// The column gather that applies a sort's permutation to a matrix. One source
// for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "gather_columns_bridge.h"

namespace {

// One thread per element of dst, column-major, so a warp stores along a column.
__global__ void gather_columns_kernel(const int32_t rows, const int64_t elements,
                                      const uint32_t *perm, const double *src, double *dst) {
  const auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) {
    return;
  }
  const int64_t r = i % rows;
  const int64_t j = i / rows;
  dst[i] = src[r + (static_cast<int64_t>(perm[j]) * rows)];
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t gather_columns(const wwr::wwrStream_t stream, const int32_t rows,
                               const int32_t cols, const uint32_t *perm, const double *src,
                               double *dst) {
  const int64_t elements = int64_t{rows} * cols;
  if (elements <= 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = static_cast<uint32_t>(math::idivup(elements, int64_t{kFourWarps}));
  gather_columns_kernel<<<blocks, kFourWarps, 0, stream>>>(rows, elements, perm, src, dst);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
