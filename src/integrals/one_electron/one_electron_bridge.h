/// @file
/// @brief The T and V launchers, declared once for the module that calls them
/// and the device TUs that define them. S has its own, in
/// overlap/overlap_bridge.h.
///
/// Included by one_electron.cpp in its GLOBAL MODULE FRAGMENT, and by
/// kinetic_launch.cu and nuclear_launch.cu directly. It has
/// to be the GMF: a declaration in a module's purview gets module linkage and
/// can never bind to a definition compiled in a plain TU, which a .cu is (wwr
/// docs/architecture.md, section 14).
///
/// wwrStream_t comes from wwr's runtime.h, whose always-on part is that one type
/// and is host-safe; it is the same type wwr.runtime_api exports, so the module
/// passes its stream straight through.
///
/// Every launcher returns the launch status as the integer value of
/// `wwrError_t` (wwrErrorInvalidValue for a class with no kernel). The enum
/// itself is only in runtime.h's device section, which a host GMF does not see;
/// the caller casts it back. Launch errors only: a kernel's own faults surface
/// at the next synchronization.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "angular_momentum.h"
#include "device_views.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief Add one class's kinetic-energy contributions into the n_ao x n_ao
/// matrix @p t, on @p stream: the generated kernel t_{la}{lb}, one thread per
/// record.
///
/// Reads @p cls's zeta, xpa, xpb, s, eb and offsets. Each value is atomically
/// added at (m, n) and at (n, m), so @p t must be zeroed before the first class
/// of a build and is symmetric after the last.
///
/// cls.n_rec must be positive, and cls.la <= cls.lb <= kMaxAngularMomentum. The
/// kernels index @p t as a 32-bit `m * n_ao + n`, so n_ao * n_ao must fit an
/// int32_t.
[[nodiscard]] int32_t kinetic_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                    int32_t n_ao, double *t);

/// @brief Add one class's nuclear-attraction contributions into the n_ao x n_ao
/// matrix @p v, on @p stream: the generated kernel v_{la}{lb}, one thread per
/// record, the @p nuclei serial inside it.
///
/// @ref kinetic_class's contract, reading @p cls's p (the product centre) in
/// place of eb, plus @p nuclei, whose n_nuc must be positive.
[[nodiscard]] int32_t nuclear_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                    const NucleiView &nuclei, int32_t n_ao, double *v);

} // namespace free_breakfast::device
