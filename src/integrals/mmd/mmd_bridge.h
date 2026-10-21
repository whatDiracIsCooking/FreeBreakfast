/// @file
/// @brief The record-order launchers, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by mmd.cpp in its GLOBAL MODULE FRAGMENT, and by mmd_launch.cu
/// directly, for the reason overlap_bridge.h gives: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU. wwrStream_t and wwrError_t come from a different wwr
/// header in each pass, as there.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief The order of a class's records by @p zeta, descending, on @p stream:
/// record perm[k] is the k-th, equal zetas in record order.
///
/// Writes key[k] = -zeta[k] and perm[k] = k, then sorts the keys ascending with
/// perm carried (src/integrals/sort's merge_sort_passes, stable). @p key,
/// @p key_alt, @p perm and @p perm_alt are @p n_rec long; the alts are touched
/// only past one tile of the sort (2816 records). No array may overlap another.
///
/// @return The first launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t mmd_order(wwr::wwrStream_t stream, int32_t n_rec, const double *zeta,
                                        double *key, double *key_alt, uint32_t *perm,
                                        uint32_t *perm_alt);

/// @brief Put @p field in the order @p perm gives, in place, on @p stream:
/// field[k] becomes the old field[perm[k]].
///
/// Copies @p field into @p tmp, then gathers it back. All three are @p n_rec
/// long, @p perm a permutation of 0 .. n_rec - 1 (@ref mmd_order's).
///
/// @return The first copy or launch status (wwrErrorInvalidValue if @p n_rec
///         is not positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t mmd_permute(wwr::wwrStream_t stream, int32_t n_rec,
                                          const uint32_t *perm, double *field, double *tmp);

/// @copydoc mmd_permute
[[nodiscard]] wwr::wwrError_t mmd_permute(wwr::wwrStream_t stream, int32_t n_rec,
                                          const uint32_t *perm, int32_t *field, int32_t *tmp);

} // namespace free_breakfast::device
