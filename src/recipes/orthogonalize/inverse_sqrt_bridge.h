/// @file
/// @brief The inverse_sqrt launcher, declared once for the module that calls it
/// and the device TU that defines it.
///
/// Included by orthogonalize.cpp in its GLOBAL MODULE FRAGMENT, and by
/// inverse_sqrt.cu directly. It has to be the GMF: a declaration in a module's
/// purview gets module linkage and can never bind to a definition compiled in a
/// plain TU, which a .cu is (wwr docs/architecture.md, section 14). The
/// `_bridge` suffix is wwr's for that: a header carrying a declaration across the
/// host/device boundary, so it must compile in both.
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

/// @brief Replace each of the first @p count elements of @p data with
/// 1 / sqrt(x), in place, on @p stream -- except elements at or below @p floor,
/// which become 0.
///
/// Flooring drops near-null and non-positive modes, and because it happens
/// before the square root, the result is never NaN or Inf. @p floor should be
/// non-negative. Instantiated for float and double.
///
/// @return The launch status. Launch errors only: the kernel's own faults
///         surface at the next synchronization.
template <typename T>
[[nodiscard]] wwr::wwrError_t inverse_sqrt(wwr::wwrStream_t stream, int32_t count, T *data,
                                           T floor);

} // namespace free_breakfast::device
