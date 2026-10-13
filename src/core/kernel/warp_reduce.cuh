/// @file
/// @brief Warp-wide register reduction over a cooperative-groups tile.
///
/// free_breakfast::device::warp_reduce folds one value per lane across a
/// kWarpSize-wide thread_block_tile under an associative binary op and returns
/// the fold in every lane: a tile.shfl_down ladder into lane 0, then one
/// tile.shfl broadcast. No shared memory and no barriers -- a shuffle is a
/// warp-level collective, so the lanes stay in step by construction.
///
/// The fold is identity-free and order-preserving: only the first @p nactive
/// lanes' values enter, combined left to right in lane order, so the op needs
/// associativity only -- no identity, no commutativity.
///
/// The value type is arithmetic: HIP's tile shuffles resolve only for vendor
/// scalar types, so a class type (a complex, an argmax pair) cannot ride a
/// shuffle portably.
///
/// Device-only: #include it into a .cu, linking
/// free_breakfast::core::kernel_headers. It has no module face.
///
/// Usage:
///   #include "warp_reduce.cuh"
///
///   auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
///   acc = free_breakfast::device::warp_reduce(tile, acc, Add{});  // every lane: the sum

#pragma once

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include <type_traits>

#include "block_params.h"
#include "device_functor.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

// The vendor cooperative-groups header, resolved per backend by WarpWraps.
#include <cooperative_groups.h>

namespace free_breakfast::device {

namespace cg = cooperative_groups;

/// @brief Fold @p v across @p warp_tile under @p op, result valid in every lane
///
/// Lanes with thread_rank() >= @p nactive contribute nothing (their @p v is
/// ignored, so no identity is needed) but must still call: every shuffle is a
/// tile-wide collective.
///
/// @tparam ParentT Deduced, so tiled_partition's result binds as-is (HIP only
///                 converts a parented tile to the <kWarpSize, void> one)
/// @tparam Op      `__device__ T operator()(T, T) const`, associative
/// @param nactive  Lanes whose @p v enters the fold; 1 <= nactive <= kWarpSize
template <typename ParentT, typename T, DeviceFunctor Op>
  requires std::is_arithmetic_v<T>
__device__ __forceinline__ T warp_reduce(cg::thread_block_tile<kWarpSize, ParentT> &warp_tile, T v,
                                         const Op op, const uint32_t nactive = kWarpSize) {
  const uint32_t lane = warp_tile.thread_rank();
  // Offsets grow, not halve: before each step lane t < nactive holds the fold of
  // the contiguous [t, min(t + offset, nactive)), and appends its neighbour's
  // [t + offset, ...) on the right. A halving ladder would pair strided lanes
  // (v0 with v16) and silently need commutativity. Every lane shuffles; only
  // the combine is guarded, so lanes >= nactive are never folded in.
  for (uint32_t offset = 1; offset < kWarpSize; offset <<= 1) {
    const T other = warp_tile.shfl_down(v, offset);
    if (lane + offset < nactive) {
      v = op(v, other);
    }
  }
  return warp_tile.shfl(v, 0);
}

} // namespace free_breakfast::device
