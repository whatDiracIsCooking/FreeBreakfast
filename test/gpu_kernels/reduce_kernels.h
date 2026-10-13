/// @file
/// @brief Host-callable launchers that run free_breakfast::device::block_reduce
/// (and through it warp_reduce) on the device, for test/gtest/reduce_gpu.cpp.
///
/// Plain C++ with no device code, as primitives_kernels.h is. Each launcher runs
/// ONE block, takes HOST arrays holding one input per thread, and writes one
/// output per thread -- so the test checks that every thread, active or not,
/// receives the fold. Every launcher returns false if any runtime call or the
/// launch failed, and leaves the outputs unspecified then.

#pragma once

#include "integral_types.h"

namespace free_breakfast::test_kernels {

/// @brief Which fold a launch applies
///
/// First and Last are associative but not commutative (the fold is the first
/// or last active input), so they catch a reduction that reorders operands or
/// lets an inactive thread's value in.
enum class FoldOp : int32_t {
  Add,
  MaxNan,
  First,
  Last,
};

/// @brief block_reduce(block, partials, v, op, nactive) over a block of
/// @p num_warps * kWarpSize threads
///
/// @p num_warps is 1, 2, 4 or 1024 / kWarpSize (the instantiations the .cu
/// carries); anything else returns false.
[[nodiscard]] bool block_reduce_on_device(FoldOp op, uint32_t num_warps, const double *in,
                                          uint32_t nactive, double *out);

/// @brief As above, over int32_t
[[nodiscard]] bool block_reduce_on_device(FoldOp op, uint32_t num_warps, const int32_t *in,
                                          uint32_t nactive, int32_t *out);

} // namespace free_breakfast::test_kernels
