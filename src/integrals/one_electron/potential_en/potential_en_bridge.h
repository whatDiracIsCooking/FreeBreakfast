/// @file
/// @brief The electron-nuclear potential launcher, declared once for the module
/// that calls it and the device TU that defines it.
///
/// Included by potential_en.cpp in its GLOBAL MODULE FRAGMENT, and by
/// potential_en_launch.cu directly. It has to be the GMF: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU, which a .cu is (wwr docs/architecture.md, section
/// 14).
///
/// wwrStream_t comes from wwr's runtime.h, whose always-on part is that one type
/// and is host-safe; it is the same type wwr.runtime_api exports, so the module
/// passes its stream straight through.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "angular_momentum.h"
#include "device_views.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief Add one class's nuclear-attraction contributions into the n_ao x n_ao
/// matrix @p v, on @p stream: the generated kernel v_{la}{lb}, one thread per
/// record, the @p nuclei serial inside it.
///
/// Reads @p cls's zeta, p_{x,y,z} (the product centre), pa_{x,y,z}, pb_{x,y,z},
/// s and offsets, and
/// @p nuclei's charges and positions. Each value is atomically added at (m, n)
/// and at (n, m), so @p v must be zeroed before the first class of a build and
/// is symmetric after the last.
///
/// cls.n_rec and nuclei.n_nuc must be positive, and cls.la <= cls.lb <=
/// kMaxAngularMomentum. The kernels index @p v as a 32-bit `m * n_ao + n`, so
/// n_ao * n_ao must fit an int32_t.
///
/// @return The launch status, as the integer value of `wwrError_t`
///         (wwrErrorInvalidValue for a class with no kernel). The enum itself is
///         only in runtime.h's device section, which a host GMF does not see;
///         the caller casts it back. Launch errors only: the kernel's own faults
///         surface at the next synchronization.
[[nodiscard]] int32_t potential_en_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                         const NucleiView &nuclei, int32_t n_ao, double *v);

} // namespace free_breakfast::device
