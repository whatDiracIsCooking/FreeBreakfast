/// @file
/// @brief The augmented Hessian's preconditioner launcher, declared once for
/// the module that calls it and the device TU that defines it.
///
/// Included by solve.cpp in its GLOBAL MODULE FRAGMENT, and by
/// ah_precondition.cu directly: a declaration in a module's purview gets
/// module linkage and can never bind to a definition compiled in a plain TU,
/// which a .cu is (see src/integrals/one_electron/overlap/overlap_bridge.h).

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

/// @brief correction_i = residual_i * (1 / d_i), d_i = theta − diagonal_i held
/// off zero at ±@p floor, for the @p n elements of one column.
///
/// The reciprocal, then the product, as the host-built 1 / d and its dgmm
/// were, so the roundings did not change when it moved to the device. Launch
/// status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t ah_precondition(wwr::wwrStream_t stream, int32_t n, double theta,
                                              double floor, const double *diagonal,
                                              const double *residual, double *correction);

} // namespace free_breakfast::device
