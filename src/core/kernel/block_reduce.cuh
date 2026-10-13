/// @file
/// @brief Block-wide reduction.
///
/// free_breakfast::device::block_reduce folds one arithmetic value per thread
/// across the block under an associative binary op and returns the fold in
/// every thread: each warp folds its lanes with warp_reduce, then every warp
/// folds the per-warp partials. The fold is identity-free: only the first
/// @p nactive threads' values enter, in thread order, so the op needs
/// associativity only. The per-warp partials go in a shared buffer the caller
/// provides, and reusing it is left to the caller.
///
/// Arithmetic values only, as for warp_reduce: HIP's tile shuffles resolve only
/// for vendor scalar types, so a class type (a complex, an argmax pair) cannot
/// be reduced here.
///
/// Device-only: #include it into a .cu, linking
/// free_breakfast::core::kernel_headers. It has no module face.
///
/// Usage:
///   #include "block_reduce.cuh"
///
///   struct Add {
///     __device__ float operator()(float a, float b) const { return a + b; }
///   };
///
///   __shared__ float partials[kNumWarps];
///   acc = block_reduce(cg::this_thread_block(), partials, acc, Add{});

#pragma once

// Standard headers first, then ours, then the wwr headers: under HIP the
// vendor runtime header must not precede libc++ (primitives.h has why), and
// primitives.h and warp_reduce.cuh each bring it in after their own standard
// headers.
#include <type_traits>

#include "block_params.h"
#include "device_functor.h"
#include "primitives.h"
#include "warp_reduce.cuh"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

// __forceinline__: nvcc pre-includes the runtime, HIP does not.
#include <runtime.h>

namespace free_breakfast::device {

/// @brief Fold arithmetic @p v across the block under @p op, through warp_reduce
///
/// Each warp folds its lanes, then every warp folds the per-warp partials in
/// warp order, so the order is thread order and associativity suffices.
/// Threads past @p nactive contribute nothing but must still call: the shuffles
/// and the barrier are collectives. One barrier per call: the caller orders any
/// earlier use of @p shared_arr before entry.
///
/// @tparam kNumWarps   Warps in the calling kernel's block: blockDim.x / kWarpSize
/// @tparam Op          `__device__ T operator()(T, T) const`, associative
/// @param block        The calling 1-D thread block, partitioned into warp tiles here
/// @param shared_arr   Shared scratch for the per-warp partials, unused when
///                     kNumWarps == 1. No thread may still read or write it on
///                     entry: block.sync() between reuses, or alternate buffers.
/// @param nactive      Threads whose @p v enters the fold;
///                     1 <= nactive <= kNumWarps * kWarpSize
template <uint32_t kNumWarps, typename T, DeviceFunctor Op>
  requires std::is_arithmetic_v<T> && NumWarps<kNumWarps>
__device__ __forceinline__ T block_reduce(const cg::thread_block &block, T (&shared_arr)[kNumWarps],
                                          const T v, const Op op,
                                          const uint32_t nactive = kNumWarps * kWarpSize) {
  auto warp_tile = cg::tiled_partition<kWarpSize>(block);
  // nactive <= kNumWarps * kWarpSize is unchecked in both branches: past it,
  // this one folds out-of-tile shuffles (a lane's own value, twice) and the
  // other reads shared_arr out of bounds.
  if constexpr (kNumWarps == 1) {
    return warp_reduce(warp_tile, v, op, nactive);
  } else {
    // NumWarps caps kNumWarps at 1024 / kWarpSize <= kWarpSize, so the second
    // level is one warp_reduce.
    const uint32_t warp_idx = warp_tile.meta_group_rank();
    const uint32_t first = warp_idx * kWarpSize;
    const uint32_t num_warps_needed = math::idivup(nactive, kWarpSize);
    // Warp-uniform branch: a warp past nactive skips its shuffles as a whole.
    if (first < nactive) {
      const uint32_t n = nactive - first < kWarpSize ? nactive - first : kWarpSize;
      const T partial = warp_reduce(warp_tile, v, op, n);
      if (warp_tile.thread_rank() == 0) {
        shared_arr[warp_idx] = partial;
      }
    }
    block.sync();
    // Every warp folds the partials itself: no broadcast, so no third barrier.
    // Lanes >= num_warps_needed read a valid slot that warp_reduce then ignores.
    const uint32_t lane = warp_tile.thread_rank();
    return warp_reduce(warp_tile, shared_arr[lane < num_warps_needed ? lane : 0], op,
                       num_warps_needed);
  }
}

} // namespace free_breakfast::device
