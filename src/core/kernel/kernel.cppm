/// @file
/// @brief Launch-shape constants (kWarpSize, kFourWarps, NumWarps) for host module code.
///
/// Usage:
///   import free_breakfast.core.kernel;
///
///   constexpr uint32_t kThreads = 4 * free_breakfast::kWarpSize;
///
/// Nothing is defined here. The names are written once, in block_params.h, so a
/// device .cu shares them; this unit GMF-includes that header and re-exports
/// the names, as primitives.cppm does. The device-only headers beside it
/// (device_functor.h, warp_reduce.cuh, block_reduce.cuh) have no module face:
/// every consumer of them today is a .cu.
///
/// The list below must name everything block_params.h defines: a name missing
/// here is still usable from a .cu but not importable.

module;

#include "block_params.h"

export module free_breakfast.core.kernel;

export namespace free_breakfast {

using free_breakfast::kFourWarps;
using free_breakfast::kWarpSize;
using free_breakfast::NumWarps;

} // namespace free_breakfast
