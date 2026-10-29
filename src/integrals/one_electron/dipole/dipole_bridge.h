/// @file
/// @brief The dipole launcher, declared once for the module that calls it and
/// the device TU that defines it.
///
/// Included by dipole.cpp in its GLOBAL MODULE FRAGMENT, and by
/// dipole_launch.cu directly; overlap_bridge.h has why it must be the GMF and
/// why wwrStream_t and wwrError_t come from a different wwr header in each pass.

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

/// @brief Add one class's dipole contributions, origin 0 and charge 1, into
/// @p mu, three n_ao x n_ao matrices x, y, z one after the other, on
/// @p stream: the generated kernel mu_{la}{lb}, one thread per record writing
/// all three.
///
/// Reads @p cls's zeta, p_{x,y,z}, pa_{x,y,z}, pb_{x,y,z}, s and offsets. Each
/// value is atomically added at (m, n) and at (n, m) of its component, so
/// @p mu must be zeroed before the first class of a build and each component
/// is symmetric after the last.
///
/// cls.n_rec must be positive, and cls.la <= cls.lb <= the largest momentum
/// kernels/dipole.h's table covers (d). The kernels index @p mu as a 32-bit
/// `(c * n_ao + m) * n_ao + n`, so 3 * n_ao * n_ao must fit an int32_t.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t
dipole_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls, int32_t n_ao, double *mu);

} // namespace free_breakfast::device
