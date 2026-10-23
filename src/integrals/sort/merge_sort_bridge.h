/// @file
/// @brief The merge-sort launcher, declared once for the device TUs that call
/// it and merge_sort.cu, which defines it.
///
/// Included by mmd_launch.cu, sort_bounds_launch.cu, exchange/pairs'
/// pairs_launch.cu and merge_sort.cu directly, and by experimental/sort/merge's merge_sort.cpp in its GLOBAL
/// MODULE FRAGMENT: a
/// declaration in a module's purview would get module linkage and never bind
/// to the definition in the .cu (see
/// src/recipes/orthogonalize/inverse_sqrt_bridge.h).

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief Sort the first @p n of @p keys ascending in IEEE-754 total order,
/// stably, on @p stream, and apply the same permutation to @p values unless it
/// is null
///
/// Same order and contract as experimental/sort/radix's radix_sort_passes: the
/// keys are moved as raw bits, -0.0 sorts before +0.0, NaNs sort by sign and
/// payload past the infinities. Data moves between the caller's arrays and the
/// @p keys_alt / @p values_alt scratch, and the result always ends in @p keys
/// and @p values. Up to 2816 keys (one tile) the scratch is never touched.
/// Every pointer is a device pointer, and none of the arrays may overlap.
///
/// @param keys_alt    Scratch, @p n elements
/// @param values      Null for a keys-only sort
/// @param values_alt  Scratch, @p n elements; ignored when @p values is null
/// @return The status of the first failed launch, or success, as the integer
///         value of `wwrError_t`. Launch errors only: a fault inside a kernel
///         surfaces at the next synchronization.
[[nodiscard]] int32_t merge_sort_passes(wwr::wwrStream_t stream, uint32_t n, double *keys,
                                        double *keys_alt, uint32_t *values, uint32_t *values_alt);

} // namespace free_breakfast::device
