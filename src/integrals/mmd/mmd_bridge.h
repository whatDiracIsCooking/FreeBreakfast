/// @file
/// @brief The record-order and record-bound launchers, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by mmd.cpp in its GLOBAL MODULE FRAGMENT, and by mmd_launch.cu
/// directly, for the reason overlap_bridge.h gives: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU. ordered_pair_bound is also included from above mmd:
/// by coulomb.cpp's GMF and by exchange/pairs' pairs_launch.cu. wwrStream_t and wwrError_t come from a different wwr
/// header in each pass, as there.

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

/// @brief Each record's Schwarz bound I = pair_bound(zeta, la + lb) into
/// @p bound, on @p stream: the generated pair_bound<la, lb>
/// (kernels/pair_bound/), one thread per record.
///
/// @p zeta and @p bound are @p n_rec long, @p bound written whole; they must
/// not overlap.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive or (la, lb) has no kernel: la <= lb <= kMaxAngularMomentum).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t mmd_bound(wwr::wwrStream_t stream, int32_t la, int32_t lb,
                                        int32_t n_rec, const double *zeta, double *bound);

/// @brief Each record's DRESSED Schwarz bound into @p bound, on @p stream: the
/// generated record_bound<cls.la, cls.lb> (kernels/record_bound/, written by
/// tools/emit_exchange_kernels.py), one thread per record, FelizJueves's
/// ao.k.record_bounds,
///
///     I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)
///
/// from the view's zeta, P - A, P - B and s, "a" its records' first shell
/// (the view's la, pa_*). The bound is ORDERED: the max runs over the first
/// shell's Cartesians, so (la, lb) and (lb, la) are different kernels with
/// different values, and @p cls may be a swapped view, la > lb with P - A and
/// P - B exchanged: mmd forms @ref PrimitivePairClass::bound_ba so, and
/// exchange/pairs its ordered classes. Only la, lb, n_rec, zeta, pa_*, pb_*
/// and s are read. @p bound is cls.n_rec long, written whole, and must not
/// overlap the view's fields; a pair whose pair factor underflows gets
/// exactly 0.0.
///
/// @return The launch status (wwrErrorInvalidValue if the view has no
///         records or no kernel: la and lb at most kMaxAngularMomentum).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t ordered_pair_bound(wwr::wwrStream_t stream,
                                                 const PrimitivePairClassView &cls, double *bound);

} // namespace free_breakfast::device
