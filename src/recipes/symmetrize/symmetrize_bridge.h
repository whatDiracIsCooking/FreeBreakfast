/// @file
/// @brief The symmetrize launchers, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by symmetrize.cpp in its GLOBAL MODULE FRAGMENT, and by
/// symmetrize.cu directly, for the reason
/// src/integrals/one_electron/overlap/overlap_bridge.h gives: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU.
///
/// The triangle is a bool here, not wwr's wwrblasFillMode_t, so the .cu needs
/// no BLAS header; the module translates.

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

/// @brief A <- (A + A^T) / 2 in place over the column-major @p n x @p n @p a
/// (leading dimension @p lda), on @p stream; (i, j) and (j, i) come out the
/// same bits.
///
/// @p n must be positive and @p lda >= @p n. Rows past @p n are neither read
/// nor written.
///
/// @return The launch status (wwrErrorInvalidValue for an @p n or @p lda out
///         of range).
[[nodiscard]] wwr::wwrError_t symmetrize(wwr::wwrStream_t stream, int32_t n, double *a,
                                         int64_t lda);

/// @brief Overwrite the strict upper triangle of the column-major @p n x @p n
/// @p a (leading dimension @p lda) with the transpose of its strict lower one
/// when @p from_lower, or the reverse, on @p stream.
///
/// The source triangle and the diagonal are not written. @p n must be positive
/// and @p lda >= @p n. Rows past @p n are neither read nor written.
///
/// @return The launch status (wwrErrorInvalidValue for an @p n or @p lda out
///         of range).
[[nodiscard]] wwr::wwrError_t mirror_triangle(wwr::wwrStream_t stream, bool from_lower, int32_t n,
                                              double *a, int64_t lda);

} // namespace free_breakfast::device
