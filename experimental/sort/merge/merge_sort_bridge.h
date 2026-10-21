/// @file
/// @brief The merge-sort launcher and the vendor reference's, declared once for
/// the module that calls them and the device TUs that define them.
///
/// Included by merge_sort.cpp in its GLOBAL MODULE FRAGMENT, and by
/// merge_sort.cu and cub_merge_sort.cu directly: a declaration in a module's purview would get module
/// linkage and never bind to the definition in the .cu (see
/// src/linalg/orthogonalize/inverse_sqrt_bridge.h).

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
/// Same order and contract as radix_sort_passes: the keys are moved as raw
/// bits, -0.0 sorts before +0.0, NaNs sort by sign and payload past the
/// infinities. Data moves between the caller's arrays and the @p keys_alt /
/// @p values_alt scratch, and the result always ends in @p keys and @p values.
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

/// @brief The bytes of temporary storage cub_merge_sort_vendor needs for @p n
/// keys, into @p bytes: the vendor's size query, which launches nothing
///
/// @return The query's status, as the integer value of `wwrError_t`
[[nodiscard]] int32_t cub_merge_sort_temp_bytes(uint32_t n, bool with_values, size_t *bytes);

/// @brief merge_sort_passes' contract and order over the vendor's stable
/// DeviceMergeSort (CUB on CUDA, hipCUB on HIP), the reference the
/// experimental sort is measured against
///
/// The vendor sorts in place, so there are no alt arrays; its scratch is all
/// in @p temp.
///
/// @param temp  Scratch, @p temp_bytes from cub_merge_sort_temp_bytes
/// @return The status of the first failed call, or success, as the integer
///         value of `wwrError_t`
[[nodiscard]] int32_t cub_merge_sort_vendor(wwr::wwrStream_t stream, uint32_t n, double *keys,
                                            uint32_t *values, void *temp, size_t temp_bytes);

} // namespace free_breakfast::device
