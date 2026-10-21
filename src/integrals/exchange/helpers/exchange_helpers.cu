// The exchange build's hand-written device steps (exchange_helpers_bridge.h):
// max|D|, the Dcd gather and the block scatter, the three things the generated
// kernel_k leaves to its launcher. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "angular_momentum.h"
#include "block_params.h"
#include "primitives.h"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;
namespace cg = cooperative_groups;

using fb::int32_t;
using fb::int64_t;
using fb::uint32_t;

// The max|D| block: the most threads a block holds.
constexpr uint32_t kMaxWarps = 1024 / fb::kWarpSize;
constexpr uint32_t kMaxThreads = kMaxWarps * fb::kWarpSize;

constexpr int64_t kMaxElements = 2147483647;

// The larger of two, with NaN winning (`a != a`): a maximum must not drop one.
struct MaxNanOp {
  __device__ double operator()(const double a, const double b) const {
    return (a != a || a > b) ? a : b;
  }
};

int32_t cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= fb::kMaxAngularMomentum;
}

// One thread per element, kFourWarps to a block.
uint32_t element_blocks(const int64_t n) {
  return static_cast<uint32_t>(fb::math::idivup(n, static_cast<int64_t>(fb::kFourWarps)));
}

// One block strides the whole matrix, each thread folding its elements from 0
// (every |D| is at least that), then the block folds the threads: one pass, one
// plain write, no atomics and no zeroing.
__global__ void __launch_bounds__(kMaxThreads)
    density_max_kernel(const int64_t n, const double *density, double *dmax) {
  __shared__ double partials[kMaxWarps];
  const MaxNanOp op{};
  double v = 0.0;
  for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
    v = op(v, fb::math::fabs(density[e]));
  }
  v = fb::device::block_reduce<kMaxWarps>(cg::this_thread_block(), partials, v, op);
  if (threadIdx.x == 0) {
    *dmax = v;
  }
}

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

// One thread per element of the block.
__global__ void scatter_block_kernel(const int64_t n, const int32_t na, const int32_t nb,
                                     const int64_t n_ao, const int32_t *off_a, const int32_t off_b,
                                     const double *block, double *k) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t width = static_cast<int64_t>(na) * nb;
  const int64_t row = e / width;
  const auto col = static_cast<int32_t>(e % width);
  const int64_t m = off_a[row] + col / nb;
  const int64_t q = off_b + col % nb;
  atomicAdd(&k[m * n_ao + q], block[e]);
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_density_max(const wwr::wwrStream_t stream, const int32_t n_ao,
                                     const double *density, double *dmax) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int64_t n = static_cast<int64_t>(n_ao) * n_ao;
  density_max_kernel<<<1, kMaxThreads, 0, stream>>>(n, density, dmax);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_gather_dcd(const wwr::wwrStream_t stream, const int32_t lc,
                                    const int32_t ld, const int32_t n_ao, const double *density,
                                    const int32_t n_c, const int32_t *off_c, const int32_t n_d,
                                    const int32_t *off_d, double *dcd) {
  if (!valid_momentum(lc) || !valid_momentum(ld) || n_ao < 1 || n_c < 1 || n_d < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nc = cartesian_count(lc);
  const int32_t nd = cartesian_count(ld);
  const int64_t n = static_cast<int64_t>(n_c) * n_d * nc * nd;
  if (n > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  gather_dcd_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(n, nc, nd, n_d, n_ao, density,
                                                                  off_c, off_d, dcd);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_scatter_block(const wwr::wwrStream_t stream, const int32_t la,
                                       const int32_t lb, const int32_t n_ao, const int32_t n_bra,
                                       const int32_t *off_a, const int32_t off_b,
                                       const double *block, double *k) {
  if (!valid_momentum(la) || !valid_momentum(lb) || n_ao < 1 || n_bra < 1 || off_b < 0) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t na = cartesian_count(la);
  const int32_t nb = cartesian_count(lb);
  const int64_t n = static_cast<int64_t>(n_bra) * na * nb;
  if (n > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  scatter_block_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(n, na, nb, n_ao, off_a, off_b,
                                                                     block, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
