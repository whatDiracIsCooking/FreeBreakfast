/// @file
/// @brief The abs_at launcher, declared once for the module that calls it and
/// the device TU that defines it.
///
/// Included by norm.cpp in its GLOBAL MODULE FRAGMENT, and by abs_at.cu
/// directly. It has to be the GMF: a declaration in a module's purview gets
/// module linkage and can never bind to a definition compiled in a plain TU,
/// which a .cu is (wwr docs/architecture.md, section 14).
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

/// @brief result[0] <- |x[(*at - 1) * incx]|, on @p stream: the magnitude of
/// the element a device-held 1-based index names.
///
/// iamax's missing back half in device pointer mode: the index lands on the
/// device, and no BLAS call reads that element back. All three pointers are
/// device memory, and *at must name an element of @p x. Instantiated for float
/// and double.
///
/// @return The launch status. Launch errors only: the kernel's own faults
///         surface at the next synchronization.
template <typename T>
[[nodiscard]] wwr::wwrError_t abs_at(wwr::wwrStream_t stream, const T *x, int32_t incx,
                                     const int32_t *at, T *result);

} // namespace free_breakfast::device
