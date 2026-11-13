/// @file
/// @brief The build target's warp width, kWarpSize, for device code.
///
/// Forked from src/core/kernel/block_params.h and trimmed to what rust/'s
/// launch wrappers use: kWarpSize. kFourWarps, kMaxThreads and NumWarps come
/// back with the first wrapper that names them. Rust never sees any of it:
/// the launch shape stays with the kernel (rust/docs/gpu_layer.md, "Launch
/// shape and dispatch").
///
/// WWR_WARP_SIZE is the configure-time constant wwr::device puts on the
/// command line (-DWWR_WARP_SIZE=32, or 64 for a CDNA build).

#pragma once

// free_breakfast::uint32_t. Standard headers only, so this is safe ahead of a
// vendor header in any includer (primitives.h has why that order matters).
#include "integral_types.h"

#ifndef WWR_WARP_SIZE
#error "WWR_WARP_SIZE is not defined: link wwr::device"
#endif

namespace free_breakfast {

/// @brief The warp/wavefront size of the build's target
inline constexpr uint32_t kWarpSize = static_cast<uint32_t>(WWR_WARP_SIZE);

static_assert(kWarpSize == 32 || kWarpSize == 64,
              "WWR_WARP_SIZE must be 32 (NVIDIA, RDNA) or 64 (CDNA)");

} // namespace free_breakfast
