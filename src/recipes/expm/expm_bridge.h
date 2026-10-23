/// @file
/// @brief The phase_scale launcher, declared once for the module that calls it
/// and the device TU that defines it.
///
/// Included by expm.cpp in its GLOBAL MODULE FRAGMENT, and by expm.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/recipes/complex_cast/complex_cast_bridge.h).
///
/// Generic in the complex type Z and its real component type R, so this header
/// names neither wwr complex type; the .cu instantiates the two pairs.

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

/// @brief v(i,j) <- u(i,j) e^{-i t w_j} over the n x n column-major @p u and
/// @p v (leading dimension n), on @p stream: U diag(e^{-i t w}), the left
/// factor of exp(tH). @p n must be positive.
///
/// @return The launch status.
template <typename Z, typename R>
[[nodiscard]] wwr::wwrError_t phase_scale(wwr::wwrStream_t stream, int32_t n, const Z *u,
                                          const R *w, R t, Z *v);

} // namespace free_breakfast::device
