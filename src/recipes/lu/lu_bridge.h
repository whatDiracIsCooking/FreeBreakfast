/// @file
/// @brief The row-interchange launcher lu_solve applies getrf's pivots with,
/// declared once for the module that calls it and the device TU that defines
/// it.
///
/// Included by lu.cpp in its GLOBAL MODULE FRAGMENT, and by lu.cu directly,
/// for the reason src/integrals/one_electron/overlap/overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU.

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

/// @brief LAPACK's laswp over all of B, on @p stream: for i = 0 .. @p n - 1 in
/// order, row i of the @p n x @p nrhs column-major @p b (leading dimension
/// @p ldb) swapped with row ipiv[i] - 1. @p ipiv is getrf's, 1-based, on the
/// device; this is B <- P B for getrf's P A = L U.
///
/// One thread per column, each walking the pivots in order, so the swaps are
/// sequential within a column as LAPACK's are.
///
/// @return The launch status (wwrErrorInvalidValue for a negative @p n or
///         @p nrhs, or an @p ldb below max(1, n)); nothing is launched when
///         either is 0. Launch errors only: the kernel's own faults surface at
///         the next synchronization.
[[nodiscard]] wwr::wwrError_t lu_row_interchanges(wwr::wwrStream_t stream, int32_t n, int32_t nrhs,
                                                  const int32_t *ipiv, double *b, int64_t ldb);

} // namespace free_breakfast::device
