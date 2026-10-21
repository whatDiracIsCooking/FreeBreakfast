/// @file
/// @brief The ket sort's launcher, declared once for the module that calls it
/// and the device TU that defines it.
///
/// Included by sort_bounds.cpp in its GLOBAL MODULE FRAGMENT, and by
/// sort_bounds_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Sort one ket class descending by Iq * Dmax, on @p stream: what
/// e_hermite's screen assumes of its kets, so that its break on the first
/// `Dmax * Ip * Iq < tau` skips only kets under tau.
///
/// Writes key[k] = -(bound[k] * dmax[k]) and perm[k] = k, sorts the keys
/// ascending with perm carried (merge_sort_passes: -0.0, a bound of zero, sorts
/// after every negative key, so last), then gathers record perm[k] of each
/// array e_hermite reads of a ket into place k of its sorted copy: zeta, P,
/// bound (Iq), dmax, and column k of each of dq's n_hermite rows.
///
/// @p bound, @p dmax and @p dq are the class's own, from coulomb_prepare.
/// @p key, @p key_alt, @p perm and @p perm_alt are cls.n_rec-long scratch; the
/// alts are touched only past one tile of the sort (2816 records). The sorted
/// copies are cls.n_rec long. @p dq and @p dq_sorted are q-major, @p n_hermite
/// x cls.n_rec, as dq writes D_q and e_hermite reads it. No array
/// may overlap another. cls.n_rec must be positive.
///
/// @return The first launch status (wwrErrorInvalidValue for no records or a
///         non-positive @p n_hermite). Launch errors only.
[[nodiscard]] wwr::wwrError_t
sort_bounds_kets(wwr::wwrStream_t stream, const PrimitivePairClassView &cls, int32_t n_hermite,
                 const double *bound, const double *dmax, const double *dq, double *key,
                 double *key_alt, uint32_t *perm, uint32_t *perm_alt, double *zeta_sorted,
                 double *p_x_sorted, double *p_y_sorted, double *p_z_sorted, double *bound_sorted,
                 double *dmax_sorted, double *dq_sorted);

} // namespace free_breakfast::device
