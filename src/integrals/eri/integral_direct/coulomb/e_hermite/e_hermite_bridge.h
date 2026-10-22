/// @file
/// @brief The [p|q] contraction's launcher, declared once for the module that
/// calls it and the device TU that defines it.
///
/// Included by e_hermite.cpp in its GLOBAL MODULE FRAGMENT, and by
/// e_hermite_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.

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

/// @brief Add the [p|q] contraction of bra class @p bra against ket class
/// @p ket into the bra's @p jp, on @p stream: the generated
/// e_hermite<bra.la + bra.lb, ket.la + ket.lb>, one warp per bra record.
///
/// @p ip is @p bra's bound (bra.bound) and @p iq, @p dq and @p dmax are
/// @p ket's, the last two from dq_class, @p dq q-major; @p jp is bra.n_rec x
/// n_hermite(bra.la + bra.lb), record-major,
/// added to atomically, so it must be zeroed before the first ket class and
/// then holds the sum over all of them. Each lane stops at its first ket with
/// `Dmax * Ip * Iq < tau`, so for any @p tau above 0 the kets (@p ket's zeta
/// and P, and @p iq, @p dq and @p dmax with them) must be sorted descending by
/// Iq * Dmax (sort_bounds_kets, into the class's ket_ fields); at @p tau = 0 the screen never fires and their order
/// is free.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t e_hermite_class(wwr::wwrStream_t stream,
                                              const PrimitivePairClassView &bra,
                                              const PrimitivePairClassView &ket, double tau,
                                              const double *ip, const double *iq, const double *dq,
                                              const double *dmax, double *jp);

} // namespace free_breakfast::device
