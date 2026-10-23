// The Dcd gather (exchange_helpers_bridge.h): the (lc, ld) shell blocks of the
// density, laid out as kb reads them. One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.

// exchange_helpers.cuh first: it orders its standard headers ahead of the
// vendor runtime header, which under HIP must not precede libc++.
#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::int64_t;

// One thread per element of Dcd, in its row-major order, so the writes
// coalesce.
__global__ void gather_dcd_kernel(const int64_t n, const int32_t nc, const int32_t nd,
                                  const int32_t n_d, const int64_t n_ao, const double *density,
                                  const int32_t *off_c, const int32_t *off_d, double *dcd) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t width = static_cast<int64_t>(nc) * nd;
  const int64_t row = e / width;
  const auto col = static_cast<int32_t>(e % width);
  const auto rank_c = static_cast<int32_t>(row / n_d);
  const auto rank_d = static_cast<int32_t>(row % n_d);
  const int64_t m = off_c[rank_c] + col / nd;
  const int64_t q = off_d[rank_d] + col % nd;
  dcd[e] = density[m * n_ao + q];
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_gather_dcd(const wwr::wwrStream_t stream, const int32_t lc,
                                    const int32_t ld, const int32_t n_ao, const double *density,
                                    const int32_t n_c, const int32_t *off_c, const int32_t n_d,
                                    const int32_t *off_d, double *dcd) {
  if (!exchange::valid_momentum(lc) || !exchange::valid_momentum(ld) || n_ao < 1 || n_c < 1 ||
      n_d < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nc = exchange::cartesian_count(lc);
  const int32_t nd = exchange::cartesian_count(ld);
  const int64_t n = static_cast<int64_t>(n_c) * n_d * nc * nd;
  if (n > exchange::kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  gather_dcd_kernel<<<exchange::element_blocks(n), kFourWarps, 0, stream>>>(
      n, nc, nd, n_d, n_ao, density, off_c, off_d, dcd);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
