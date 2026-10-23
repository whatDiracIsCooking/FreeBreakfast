/// @file
/// @brief The kinetic-energy launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by kinetic.cpp in its GLOBAL MODULE FRAGMENT, and by
/// kinetic_launch.cu directly. It has to be the GMF: a declaration in a module's
/// purview gets module linkage and can never bind to a definition compiled in a
/// plain TU, which a .cu is (wwr docs/architecture.md, section 14).
///
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// overlap_bridge.h explains, so the module passes its stream in and takes
/// the status back with no cast.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "angular_momentum.h"
#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Add one class's kinetic-energy contributions into the n_ao x n_ao
/// matrix @p t, on @p stream: the generated kernel t_{la}{lb}, one thread per
/// record.
///
/// Reads @p cls's zeta, pa_{x,y,z}, pb_{x,y,z}, s, eb and offsets. Each value is
/// atomically added at (m, n) and at (n, m), so @p t must be zeroed before the first class
/// of a build and is symmetric after the last.
///
/// cls.n_rec must be positive, and cls.la <= cls.lb <= kMaxAngularMomentum. The
/// kernels index @p t as a 32-bit `m * n_ao + n`, so n_ao * n_ao must fit an
/// int32_t.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t
kinetic_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls, int32_t n_ao, double *t);

} // namespace free_breakfast::device
