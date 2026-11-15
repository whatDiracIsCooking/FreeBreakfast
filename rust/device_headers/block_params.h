/// @file
/// @brief The build target's warp width, kWarpSize, the block kFourWarps,
/// the block limit kMaxThreads and the warps-per-block concept NumWarps, for
/// device code.
///
/// Forked from src/core/kernel/block_params.h and trimmed to what rust/'s
/// launch wrappers use: kWarpSize, kFourWarps, kMaxThreads and NumWarps,
/// their definitions unchanged. Rust never sees any of it:
/// the launch shape stays with the kernel (rust/docs/gpu_layer.md, "Launch
/// shape and dispatch").
///
/// WWR_WARP_SIZE is the configure-time constant wwr::device puts on the
/// command line (-DWWR_WARP_SIZE=32, or 64 for a CDNA build).

#pragma once

#include <concepts>

// free_breakfast::uint32_t and uint64_t. Standard headers only, so this is safe ahead of a
// vendor header in any includer (primitives.h has why that order matters).
#include "integral_types.h"

#ifndef WWR_WARP_SIZE
#error "WWR_WARP_SIZE is not defined: link wwr::device"
#endif

namespace free_breakfast {

/// @brief The warp/wavefront size of the build's target
inline constexpr uint32_t kWarpSize = static_cast<uint32_t>(WWR_WARP_SIZE);

/// @brief Four warps' worth of threads: 128 on NVIDIA and RDNA, 256 on CDNA
///
/// The block size of a one-thread-per-item launch, such as the one-electron
/// integral launchers: a whole number of warps keeps every block's lanes in step.
inline constexpr uint32_t kFourWarps = 4 * kWarpSize;

/// @brief The most threads a block may hold: 1024, on every target either
/// backend builds for
inline constexpr uint32_t kMaxThreads = 1024;

static_assert(kWarpSize == 32 || kWarpSize == 64,
              "WWR_WARP_SIZE must be 32 (NVIDIA, RDNA) or 64 (CDNA)");
static_assert(kMaxThreads % kWarpSize == 0, "a full block must be whole warps");

/// @brief A warps-per-block count @p N: integral, >= 1, a power of two, and at
/// most kMaxThreads threads once multiplied by kWarpSize
///
/// Constrains a value, not a type: `template <auto W> requires NumWarps<W>`.
/// The bound is checked as N <= kMaxThreads / kWarpSize so a huge N cannot overflow
/// the product, and in uint64_t so a 64-bit N (1ull << 32, say) is not
/// truncated to 0 and let through; N >= 1 has already ruled out negatives.
template <auto N>
concept NumWarps = std::integral<decltype(N)> && (N >= 1) && ((N & (N - 1)) == 0) &&
                   (static_cast<uint64_t>(N) <= kMaxThreads / kWarpSize);

} // namespace free_breakfast
