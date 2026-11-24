/// @file
/// @brief The trsm diagonal-block launcher and its block shape, declared once
/// for the module that calls it and the device TU that defines it.
///
/// Included by trsm.cpp in its GLOBAL MODULE FRAGMENT, and by trsm.cu
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

/// @brief The order of the diagonal blocks trsm solves in shared memory, and
/// so the inner dimension of its gemm updates. A block is 64^2 doubles,
/// 32 KiB.
inline constexpr int32_t kTrsmBlock = 64;

/// @brief The right-hand sides one thread block of the diagonal-block kernel
/// takes: its tile is kTrsmBlock x kTrsmColumns, 24 KiB of doubles. With the
/// packed triangle (16.25 KiB) and its diagonal's reciprocals the kernel holds
/// 40.75 KiB of shared memory, under CUDA's 48 KiB static limit and the
/// 64 KiB of LDS under HIP.
inline constexpr int32_t kTrsmColumns = 48;

/// @brief The threads of the kernel that share one right-hand side, each
/// taking every kTrsmLanes-th row of an update.
inline constexpr int32_t kTrsmLanes = 4;

/// @brief The kernel's thread block: kTrsmLanes threads to each of its
/// kTrsmColumns columns, 192, whole warps at a warp of 32 or 64.
inline constexpr uint32_t kTrsmThreads = kTrsmColumns * kTrsmLanes;

/// @brief Solve M X = s B in place for one @p jb x @p jb triangular diagonal
/// block M and @p ncols right-hand sides, on @p stream.
///
/// M(p, q) is a[q + p lda] when @p a_trans, else a[p + q lda]; it is upper
/// triangular when @p upper, else lower, and only that triangle is read. With
/// @p unit its diagonal is taken as 1 and not read. B(r, c) is b[c + r ldb]
/// when @p b_trans, else b[r + c ldb], and X overwrites it. With @p s zero, X
/// is zero and neither M nor B is read (BLAS's alpha = 0).
///
/// @p jb must be in 1..kTrsmBlock and @p ncols positive.
///
/// @return The launch status (wwrErrorInvalidValue for an argument out of
///         range). Launch errors only: the kernel's own faults surface at the
///         next synchronization.
[[nodiscard]] wwr::wwrError_t trsm_diagonal_block(wwr::wwrStream_t stream, int32_t jb,
                                                  int32_t ncols, float s, const float *a,
                                                  int64_t lda, bool a_trans, bool upper, bool unit,
                                                  float *b, int64_t ldb, bool b_trans);

/// @brief The double overload of the one above.
[[nodiscard]] wwr::wwrError_t trsm_diagonal_block(wwr::wwrStream_t stream, int32_t jb,
                                                  int32_t ncols, double s, const double *a,
                                                  int64_t lda, bool a_trans, bool upper, bool unit,
                                                  double *b, int64_t ldb, bool b_trans);

} // namespace free_breakfast::device
