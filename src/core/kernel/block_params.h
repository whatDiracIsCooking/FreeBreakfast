/// @file
/// @brief Launch-shape constants and constraints (kWarpSize, kFourWarps,
/// kMaxThreads, NumWarps), written once for host and device code.
///
/// A plain header, as primitives.h is: block sizing is wanted by device
/// translation units, which cannot `import`, and by host code sizing the launch.
/// kernel.cppm GMF-includes it and re-exports the names.
///
/// WWR_WARP_SIZE is a configure-time constant that wwr::device puts on the
/// command line (-DWWR_WARP_SIZE=32, or 64 for a CDNA build), in a host compile
/// and a device pass alike. This header reads the macro rather than including
/// wwr's runtime.h for it: runtime.h defines the macro only inside its
/// device-pass section, and in a host compile would bring the vendor runtime API
/// header into a module's GMF for nothing.
///
/// Link free_breakfast::core::kernel_headers from device code.

#pragma once

// Standard headers before any vendor header (primitives.h has why); this one
// includes none, so it is safe ahead of them in any includer.
#include <concepts>

// free_breakfast::uint32_t / uint64_t.
#include "integral_types.h"

#ifndef WWR_WARP_SIZE
#error "WWR_WARP_SIZE is not defined: link wwr::device (free_breakfast::core::kernel_headers does)"
#endif

namespace free_breakfast {

/// @brief The warp/wavefront size of the build's target
///
/// `inline` gives it external linkage, which kernel.cppm needs to export it (a
/// plain namespace-scope constexpr is internal).
inline constexpr uint32_t kWarpSize = static_cast<uint32_t>(WWR_WARP_SIZE);

/// @brief Four warps' worth of threads: 128 on NVIDIA and RDNA, 256 on CDNA
///
/// The block size of a one-thread-per-item launch, such as the one-electron
/// integral launchers: a whole number of warps keeps every block's lanes in step.
inline constexpr uint32_t kFourWarps = 4 * kWarpSize;

/// @brief The most threads a block may hold: 1024, on every target either
/// backend builds for
///
/// The block size of a one-block launch that strides a whole array, and the
/// bound NumWarps holds a warps-per-block count to.
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
