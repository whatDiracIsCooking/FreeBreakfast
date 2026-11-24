// The device half of free_breakfast.recipes.lu: getrf's row interchanges
// applied to B. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "lu_bridge.h"

namespace free_breakfast::device {

namespace {

// Column c of B, its rows swapped in pivot order. The pivots are read by
// every thread alike, so they broadcast; the swaps within a column must stay
// in order, since a later pivot may name a row an earlier one moved.
__global__ void lu_row_interchanges_kernel(const int32_t n, const int32_t nrhs,
                                           const int32_t *__restrict__ ipiv, double *__restrict__ b,
                                           const int64_t ldb) {
  const int64_t c = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (c >= nrhs) {
    return;
  }
  double *col = b + (c * ldb);
  for (int32_t i = 0; i < n; ++i) {
    const int32_t p = ipiv[i] - 1;
    if (p != i) {
      const double t = col[i];
      col[i] = col[p];
      col[p] = t;
    }
  }
}

} // namespace

wwr::wwrError_t lu_row_interchanges(const wwr::wwrStream_t stream, const int32_t n,
                                    const int32_t nrhs, const int32_t *ipiv, double *b,
                                    const int64_t ldb) {
  if (n < 0 || nrhs < 0 || ldb < (n > 1 ? n : 1)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0 || nrhs == 0) {
    return wwr::wwrSuccess;
  }
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(nrhs), kFourWarps);
  lu_row_interchanges_kernel<<<blocks, kFourWarps, 0, stream>>>(n, nrhs, ipiv, b, ldb);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
