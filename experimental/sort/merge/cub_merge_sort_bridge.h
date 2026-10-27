/// @file
/// @brief The vendor reference's launchers, declared once for the module that
/// calls them and cub_merge_sort.cu, which defines them.
///
/// Included by merge_sort.cpp in its GLOBAL MODULE FRAGMENT, and by
/// cub_merge_sort.cu directly: a declaration in a module's purview would get
/// module linkage and never bind to the definition in the .cu (see
/// src/recipes/orthogonalize/inverse_sqrt_bridge.h). merge_sort itself is
/// src/recipes/sort's merge_sort.cu, declared by its merge_sort_bridge.h.
/// wwrError_t comes from a different wwr header in each pass, as
/// src/integrals/one_electron/overlap/overlap_bridge.h explains.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief The bytes of temporary storage cub_merge_sort_vendor needs for @p n
/// keys, into @p bytes: the vendor's size query, which launches nothing
///
/// @return The query's status
[[nodiscard]] wwr::wwrError_t cub_merge_sort_temp_bytes(uint32_t n, bool with_values,
                                                        size_t *bytes);

/// @brief merge_sort_passes' contract and order over the vendor's stable
/// DeviceMergeSort (CUB on CUDA, hipCUB on HIP), the reference the
/// experimental sort is measured against
///
/// The vendor sorts in place, so there are no alt arrays; its scratch is all
/// in @p temp.
///
/// @param temp  Scratch, @p temp_bytes from cub_merge_sort_temp_bytes
/// @return The status of the first failed call, or success
[[nodiscard]] wwr::wwrError_t cub_merge_sort_vendor(wwr::wwrStream_t stream, uint32_t n,
                                                    double *keys, uint32_t *values, void *temp,
                                                    size_t temp_bytes);

} // namespace free_breakfast::device
