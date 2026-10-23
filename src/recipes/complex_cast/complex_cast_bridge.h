/// @file
/// @brief The split_planes / merge_planes launchers, declared once for the
/// module that calls them and the device TU that defines them.
///
/// Included by complex_cast.cpp in its GLOBAL MODULE FRAGMENT, and by
/// complex_cast.cu directly. It has to be the GMF: a declaration in a module's
/// purview gets module linkage and can never bind to a definition compiled in a
/// plain TU, which a .cu is (wwr docs/architecture.md, section 14).
///
/// The launchers are generic in the complex type Z and its real component type
/// R, so this header names neither wwr complex type; the caller spells R as
/// wwr::ComplexToRealType<Z>, and the .cu instantiates the two pairs.
///
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// src/integrals/one_electron/overlap/overlap_bridge.h explains, so the module
/// passes its stream in and takes the status back with no cast.

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

/// @brief re(i,j) <- Re(a(i,j)), im(i,j) <- Im(a(i,j)) over the column-major
/// @p rows x @p cols block, on @p stream.
///
/// @p a has leading dimension @p lda, both planes @p ldp; padding rows are
/// neither read nor written. @p rows and @p cols must be positive.
///
/// @return The launch status.
template <typename Z, typename R>
[[nodiscard]] wwr::wwrError_t split_planes(wwr::wwrStream_t stream, int32_t rows, int32_t cols,
                                           const Z *a, int32_t lda, R *re, R *im, int32_t ldp);

/// @brief c(i,j) <- (re(i,j), im(i,j)) over the column-major @p rows x @p cols
/// block, on @p stream; split_planes' inverse.
///
/// Both planes have leading dimension @p ldp, @p c has @p ldc; padding rows are
/// neither read nor written. @p rows and @p cols must be positive.
///
/// @return The launch status.
template <typename Z, typename R>
[[nodiscard]] wwr::wwrError_t merge_planes(wwr::wwrStream_t stream, int32_t rows, int32_t cols,
                                           const R *re, const R *im, int32_t ldp, Z *c,
                                           int32_t ldc);

} // namespace free_breakfast::device
