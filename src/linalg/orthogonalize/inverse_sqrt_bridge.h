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
/// wwrStream_t comes from wwr's runtime.h, whose always-on part is that one type
/// and is host-safe; it is the same type wwr.runtime_api exports, so the module
/// passes its stream straight through.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief Replace each of the first @p count elements of @p data with
/// 1 / sqrt(x), in place, on @p stream -- except elements at or below @p floor,
/// which become 0.
///
/// Flooring drops near-null and non-positive modes, and because it happens
/// before the square root, the result is never NaN or Inf. @p floor should be
/// non-negative. Instantiated for float and double.
///
/// @return The launch status, as the integer value of `wwrError_t`. The enum
///         itself is only in runtime.h's device section, which a host GMF does
///         not see; the caller casts it back. Launch errors only: the kernel's
///         own faults surface at the next synchronization.
template <typename T>
[[nodiscard]] int32_t inverse_sqrt(wwr::wwrStream_t stream, int32_t count, T *data, T floor);

} // namespace free_breakfast::device
