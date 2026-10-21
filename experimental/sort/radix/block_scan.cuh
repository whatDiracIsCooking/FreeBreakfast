/// @file
/// @brief Block-wide exclusive prefix sum of one uint32_t per thread.
///
/// The scan counterpart of core/kernel's block_reduce, built the same way: a
/// warp-level shuffle ladder per warp, the per-warp totals through a shared
/// buffer the caller provides, and one barrier per call. It lives beside the
/// radix sort rather than in core/kernel while that sort is its only user;
/// promoting it means generalising the value type and op the way block_reduce
/// is generalised.
///
/// Device-only: #include it into a .cu, linking
/// free_breakfast::core::kernel_headers.

#pragma once

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <cooperative_groups.h>

// __forceinline__: nvcc pre-includes the runtime, HIP does not.
#include <runtime.h>

namespace free_breakfast::device {

namespace cg = cooperative_groups;

/// @brief Exclusive prefix sum of @p v across the block, in thread order
///
/// Thread t receives the sum of the values of threads 0 .. t-1 (0 in thread 0),
/// and every thread receives the block's total in @p total. Every thread must
/// call: the shuffles and the barrier are collectives. A thread with nothing to
/// contribute passes 0. The sum wraps modulo 2^32.
///
/// @tparam kNumWarps   Warps in the calling kernel's block: blockDim.x / kWarpSize
/// @param block        The calling 1-D thread block
/// @param shared_arr   Shared scratch for the per-warp totals, unused when
///                     kNumWarps == 1. As for block_reduce: block.sync()
///                     between two calls that share it.
/// @param total        Out: the sum over the whole block, in every thread
template <uint32_t kNumWarps>
  requires NumWarps<kNumWarps>
__device__ __forceinline__ uint32_t block_exclusive_sum(const cg::thread_block &block,
                                                        uint32_t (&shared_arr)[kNumWarps],
                                                        const uint32_t v, uint32_t &total) {
  auto warp_tile = cg::tiled_partition<kWarpSize>(block);
  const uint32_t lane = warp_tile.thread_rank();

  // Hillis-Steele inclusive scan within the warp. Every lane shuffles; only the
  // add is guarded, so the shuffle stays a full-tile collective.
  uint32_t inclusive = v;
  for (uint32_t offset = 1; offset < kWarpSize; offset <<= 1) {
    const uint32_t other = warp_tile.shfl_up(inclusive, offset);
    if (lane >= offset) {
      inclusive += other;
    }
  }

  if constexpr (kNumWarps == 1) {
    total = warp_tile.shfl(inclusive, kWarpSize - 1);
    return inclusive - v;
  } else {
    const uint32_t warp_idx = warp_tile.meta_group_rank();
    if (lane == kWarpSize - 1) {
      shared_arr[warp_idx] = inclusive;
    }
    block.sync();
    // Every warp scans the per-warp totals itself, as block_reduce folds them:
    // no broadcast, so no second barrier. NumWarps caps kNumWarps at
    // kWarpSize, so one warp-wide ladder covers them.
    const uint32_t warp_total = lane < kNumWarps ? shared_arr[lane] : 0;
    uint32_t warp_inclusive = warp_total;
    for (uint32_t offset = 1; offset < kWarpSize; offset <<= 1) {
      const uint32_t other = warp_tile.shfl_up(warp_inclusive, offset);
      if (lane >= offset) {
        warp_inclusive += other;
      }
    }
    const uint32_t warp_prefix = warp_tile.shfl(warp_inclusive - warp_total, warp_idx);
    total = warp_tile.shfl(warp_inclusive, kNumWarps - 1);
    return warp_prefix + inclusive - v;
  }
}

} // namespace free_breakfast::device
