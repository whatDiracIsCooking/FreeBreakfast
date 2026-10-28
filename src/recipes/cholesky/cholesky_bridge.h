/// @file
/// @brief The Cholesky diagonal-block launcher and its block order, declared
/// once for the module that calls it and the device TU that defines it.
///
/// Included by cholesky.cpp in its GLOBAL MODULE FRAGMENT, and by cholesky.cu
/// directly, for the reason src/integrals/one_electron/overlap/overlap_bridge.h
/// gives: a declaration in a module's purview gets module linkage and can never
/// bind to a definition compiled in a plain TU.

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

/// @brief The order of the diagonal blocks cholesky factors in shared memory,
/// and so the width of its trsm and syrk panels. A block is 64^2 doubles,
/// 32 KiB, under CUDA's 48 KiB static limit.
inline constexpr int32_t kCholeskyBlock = 64;

/// @brief Factor the @p jb x @p jb diagonal block at @p a (leading dimension
/// @p lda) in place, A = L L^T in its lower triangle, on @p stream: one block,
/// the matrix in shared memory.
///
/// @p first is the block's first column in the whole matrix. If @p info is
/// already non-zero on entry (an earlier block met a minor that is not
/// positive), the block is left as it is. Otherwise, at the first pivot that is
/// not positive (or is NaN), column k of the block, @p info becomes
/// first + k + 1, the 1-based leading minor of the whole matrix, and the block
/// is left part-factored. The strict upper triangle is never written.
/// @p jb must be in 1..kCholeskyBlock.
///
/// @return The launch status (wwrErrorInvalidValue for a @p jb out of range).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t cholesky_diagonal_block(wwr::wwrStream_t stream, int32_t jb,
                                                      int32_t first, double *a, int64_t lda,
                                                      int32_t *info);

} // namespace free_breakfast::device
