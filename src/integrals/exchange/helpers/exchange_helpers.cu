// The exchange build's hand-written device steps (exchange_helpers_bridge.h):
// max|D|, the Dcd gather, the block table, and the shell starts and mirror,
// the things the generated kb leaves to its launcher. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.

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

// The block-table block: kFourWarps threads.
constexpr uint32_t kTableWarps = fb::kFourWarps / fb::kWarpSize;

constexpr int64_t kMaxElements = 2147483647;

// The larger of two, with NaN winning (`a != a`): a maximum must not drop one.
struct MaxNanOp {
  __device__ double operator()(const double a, const double b) const {
    return (a != a || a > b) ? a : b;
  }
};

struct AddOp {
  __device__ int32_t operator()(const int32_t a, const int32_t b) const { return a + b; }
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

// The first ket group whose first AO is at least off: with the symmetric
// sweep, a bra group at off takes the ket groups from it on. The groups are in
// basis order, so their first AOs rise and a binary search finds it.
__device__ int32_t first_ket_group(const int32_t n_b, const int32_t *b_start, const int32_t *b_off,
                                   const int32_t off) {
  int32_t lo = 0;
  int32_t hi = n_b;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (b_off[b_start[mid]] < off) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// One block per bra group ga. Its blocks start after every earlier bra group's,
// n_b - first(g) each, which the block's threads sum, and its threads then
// write its own, one per ket group from first(ga) on.
__global__ void __launch_bounds__(fb::kFourWarps)
    blocks_kernel(const int32_t *a_start, const int32_t *a_off, const int32_t n_b,
                  const int32_t *b_start, const int32_t *b_off, const bool symmetric,
                  const int32_t n_blk, int32_t *blocks) {
  __shared__ int32_t partials[kTableWarps];
  const auto ga = static_cast<int32_t>(blockIdx.x);
  const auto first = [&](const int32_t g) {
    return symmetric ? first_ket_group(n_b, b_start, b_off, a_off[a_start[g]]) : 0;
  };
  int32_t before = 0;
  for (auto g = static_cast<int32_t>(threadIdx.x); g < ga; g += static_cast<int32_t>(blockDim.x)) {
    before += n_b - first(g);
  }
  const int32_t base =
      fb::device::block_reduce<kTableWarps>(cg::this_thread_block(), partials, before, AddOp{});
  const int32_t f = first(ga);
  const int32_t bra0 = a_start[ga];
  for (int32_t gb = f + static_cast<int32_t>(threadIdx.x); gb < n_b;
       gb += static_cast<int32_t>(blockDim.x)) {
    const int32_t blk = base + gb - f;
    const int32_t ket0 = b_start[gb];
    blocks[blk] = bra0;
    blocks[n_blk + blk] = a_start[ga + 1];
    blocks[2 * n_blk + blk] = ket0;
    blocks[3 * n_blk + blk] = b_start[gb + 1];
    blocks[4 * n_blk + blk] = a_off[bra0];
    blocks[5 * n_blk + blk] = b_off[ket0];
  }
}

// One thread per AO of the momentum's shells, by rank.
__global__ void shell_starts_kernel(const int32_t n, const int32_t nx, const int32_t *off,
                                    int32_t *start) {
  const auto e = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < n) {
    const int32_t o = off[e / nx];
    start[o + e % nx] = o;
  }
}

// One thread per element of K; those of a lower shell block read their
// transpose, which is in an upper one, so no element is both read and written.
__global__ void mirror_kernel(const int64_t n, const int64_t n_ao, const int32_t *start,
                              double *k) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t m = e / n_ao;
  const int64_t q = e % n_ao;
  if (q < start[m]) {
    k[e] = k[q * n_ao + m];
  }
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

wwr::wwrError_t exchange_blocks(const wwr::wwrStream_t stream, const int32_t n_a,
                                const int32_t *a_start, const int32_t *a_off, const int32_t n_b,
                                const int32_t *b_start, const int32_t *b_off, const bool symmetric,
                                const int32_t n_blk, int32_t *blocks) {
  if (n_a < 1 || n_b < 1 || n_blk < 1 ||
      static_cast<int64_t>(n_blk) * kExchangeBlockFields > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  blocks_kernel<<<static_cast<uint32_t>(n_a), kFourWarps, 0, stream>>>(
      a_start, a_off, n_b, b_start, b_off, symmetric, n_blk, blocks);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_shell_starts(const wwr::wwrStream_t stream, const int32_t l,
                                      const int32_t n_s, const int32_t *off, int32_t *start) {
  if (!valid_momentum(l) || n_s < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nx = cartesian_count(l);
  const int64_t n = static_cast<int64_t>(n_s) * nx;
  if (n > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  shell_starts_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(static_cast<int32_t>(n), nx,
                                                                    off, start);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_mirror(const wwr::wwrStream_t stream, const int32_t n_ao,
                                const int32_t *start, double *k) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int64_t n = static_cast<int64_t>(n_ao) * n_ao;
  mirror_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(n, n_ao, start, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
