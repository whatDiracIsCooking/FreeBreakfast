/// @file
/// @brief The Aberth-Ehrlich launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by polynomial_roots.cpp in its GLOBAL MODULE FRAGMENT, and by
/// aberth.cu directly. It has to be the GMF: a declaration in a module's
/// purview gets module linkage and can never bind to a definition compiled in a
/// plain TU, which a .cu is (wwr docs/architecture.md, section 14).
///
/// The launcher is generic in the coefficient type T, the root type CT and
/// their real type R, so this header names no wwr complex type; the .cu
/// instantiates the four combinations. It is unconstrained because a
/// requires-clause enters the mangled name, which nvcc and clang spell
/// differently.
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

/// @brief Every root of @p batch degree-@p n polynomials, one warp each and one
/// lane per root, on @p stream.
///
/// Polynomial b is coeffs[b*(n+1) + k] = a_k (ascending); its roots land in
/// roots[b*n, b*n + n) and its iteration count, or -1, in iters[b]. Requires
/// 1 <= n <= kWarpSize and batch >= 1.
///
/// @return The launch status.
template <typename T, typename CT, typename R>
[[nodiscard]] wwr::wwrError_t aberth(wwr::wwrStream_t stream, int32_t n, int32_t batch,
                                     const T *coeffs, CT *roots, int32_t *iters, R tol,
                                     int32_t max_iter);

} // namespace free_breakfast::device
