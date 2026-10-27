/// @file
/// @brief The column-gather launcher, declared once for the module that calls
/// it and gather_columns.cu, which defines it.
///
/// Included by sort.cpp in its GLOBAL MODULE FRAGMENT, and by gather_columns.cu
/// directly: a declaration in a module's purview would get module linkage and
/// never bind to the definition in the .cu (see
/// src/recipes/orthogonalize/inverse_sqrt_bridge.h). wwrError_t comes from a
/// different wwr header in each pass, as
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

/// @brief Column j of @p dst = column perm[j] of @p src, for @p rows x @p cols
/// column-major matrices, on @p stream
///
/// One thread per element of @p dst, so its stores coalesce. Every pointer is
/// a device pointer, @p dst must not overlap @p src, and every perm[j] must be
/// below @p cols.
///
/// @return The launch's status: a fault inside the kernel surfaces at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t gather_columns(wwr::wwrStream_t stream, int32_t rows, int32_t cols,
                                             const uint32_t *perm, const double *src, double *dst);

} // namespace free_breakfast::device
