// The block table of one kb launch and its factors (exchange_helpers_bridge.h):
// each (bra group, ket group) block's records and AO offsets, and, a second
// launch of the same walk, its density factor.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;
namespace cg = cooperative_groups;

using fb::int32_t;
using fb::int64_t;
using fb::device::exchange::distance;
using fb::device::exchange::kMaxElements;
using fb::device::exchange::kTableWarps;

struct AddOp {
  __device__ int32_t operator()(const int32_t a, const int32_t b) const { return a + b; }
};

// The first ket group whose first AO is at least off: a bra group at off takes
// the ket groups from it on. The groups are in basis order, so their first AOs
// rise and a binary search finds it.
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

// F_ab of bra group ga against ket group gb: see exchange_factors.
__device__ double block_factor(const int32_t n_a, const double *a_geom, const int32_t ga,
                               const int32_t n_b, const double *b_geom, const int32_t gb,
                               const double *decay) {
  const double dm = decay[0];
  const double ra = a_geom[3 * n_a + ga];
  const double rb = b_geom[3 * n_b + gb];
  if (ra < 0.0 || rb < 0.0) {
    return dm;
  }
  const double d = distance(a_geom[ga], a_geom[n_a + ga], a_geom[2 * n_a + ga], b_geom[gb],
                            b_geom[n_b + gb], b_geom[2 * n_b + gb]) -
                   ra - rb;
  if (!(d > 2.0 * fb::device::kExchangeSameCentre)) {
    return dm;
  }
  return fb::math::fmin(dm, decay[2] * fb::math::exp(-decay[1] * d) *
                                (1.0 + fb::device::kExchangeDecayMargin));
}

// One block per bra group ga. Its blocks start after every earlier bra group's,
// n_b - first(g) each, which the block's threads sum, and its threads then
// visit its own, one per ket group from first(ga) on: each block's table
// entry when blocks is not null, and its factor when factor is not null. The
// table and the factors are two launches (exchange_blocks, exchange_factors)
// of this one walk, so they agree on the order.
__global__ void __launch_bounds__(fb::kFourWarps)
    blocks_kernel(const int32_t n_a, const int32_t *a_start, const int32_t *a_off,
                  const double *a_geom, const int32_t n_b, const int32_t *b_start,
                  const int32_t *b_off, const double *b_geom, const double *decay,
                  const int32_t n_blk, int32_t *blocks, double *factor) {
  __shared__ int32_t partials[kTableWarps];
  const auto ga = static_cast<int32_t>(blockIdx.x);
  const auto first = [&](const int32_t g) {
    return first_ket_group(n_b, b_start, b_off, a_off[a_start[g]]);
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
    if (blocks != nullptr) {
      const int32_t ket0 = b_start[gb];
      blocks[blk] = bra0;
      blocks[n_blk + blk] = a_start[ga + 1];
      blocks[2 * n_blk + blk] = ket0;
      blocks[3 * n_blk + blk] = b_start[gb + 1];
      blocks[4 * n_blk + blk] = a_off[bra0];
      blocks[5 * n_blk + blk] = b_off[ket0];
    }
    if (factor != nullptr) {
      factor[blk] = block_factor(n_a, a_geom, ga, n_b, b_geom, gb, decay);
    }
  }
}

// The checks exchange_blocks and exchange_factors share.
bool valid_table(const int32_t n_a, const int32_t n_b, const int32_t n_blk) {
  return n_a >= 1 && n_b >= 1 && n_blk >= 1 &&
         static_cast<int64_t>(n_blk) * fb::device::kExchangeBlockFields <= kMaxElements;
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_blocks(const wwr::wwrStream_t stream, const int32_t n_a,
                                const int32_t *a_start, const int32_t *a_off, const int32_t n_b,
                                const int32_t *b_start, const int32_t *b_off, const int32_t n_blk,
                                int32_t *blocks) {
  if (!valid_table(n_a, n_b, n_blk)) {
    return wwr::wwrErrorInvalidValue;
  }
  blocks_kernel<<<static_cast<uint32_t>(n_a), kFourWarps, 0, stream>>>(
      n_a, a_start, a_off, nullptr, n_b, b_start, b_off, nullptr, nullptr, n_blk, blocks, nullptr);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_factors(const wwr::wwrStream_t stream, const int32_t n_a,
                                 const int32_t *a_start, const int32_t *a_off, const double *a_geom,
                                 const int32_t n_b, const int32_t *b_start, const int32_t *b_off,
                                 const double *b_geom, const double *decay, const int32_t n_blk,
                                 double *factor) {
  if (!valid_table(n_a, n_b, n_blk)) {
    return wwr::wwrErrorInvalidValue;
  }
  blocks_kernel<<<static_cast<uint32_t>(n_a), kFourWarps, 0, stream>>>(
      n_a, a_start, a_off, a_geom, n_b, b_start, b_off, b_geom, decay, n_blk, nullptr, factor);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
