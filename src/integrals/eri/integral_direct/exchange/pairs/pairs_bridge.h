/// @file
/// @brief The ordered classes' order, gather and Dcd-row launchers, declared
/// once for the module that calls them and the device TU that defines them.
///
/// Included by pairs.cpp in its GLOBAL MODULE FRAGMENT, and by
/// pairs_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.
///
/// Both launchers read an ordered class straight from the mmd class it is
/// drawn from, through the mmd class's view, by SLOT: slot s is one
/// orientation of one mmd record. For an ordered (la, lb) of an mmd (la, lb)
/// with la < lb, slot r is record r as stored; of an mmd (lb, la), record r
/// swapped; and for la == lb, slot 2r is record r as stored and slot 2r + 1 it
/// swapped, a slot that does not exist for a diagonal record (off_a == off_b),
/// which has one orientation only. So a class has `from.n_rec` slots, twice
/// that for la == lb.
///
/// Every pointer is device memory, every launch is asynchronous on @p stream,
/// and the launchers return launch errors only.

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

/// @brief The order of the ordered class (@p la, lb) of the mmd class
/// @p from, on @p stream: slot perm[k] is its k-th record, for k below the
/// class's record count, grouped by first shell in basis order (group g the
/// first shell of rank g among the shells of momentum @p la), each group in
/// descending dressed bound (mmd's bound_ab as stored, bound_ba swapped),
/// equal bounds in slot order; and the groups' bounds into @p group_start.
///
/// Two stable sorts (src/recipes/sort's merge_sort_passes): every slot by
/// -bound, then the result by its first shell's rank, a slot that does not
/// exist by @p n_groups, so it lands past every record. @p group_start, of
/// @p n_groups + 1 entries, is then each group's first record, by bisection
/// over the sorted ranks, its last entry the record count. @p key,
/// @p key_alt, @p perm and @p perm_alt are one per slot; the alts are touched
/// only past one tile of the sort (2816 slots). No array may overlap another,
/// or @p from's fields.
///
/// @return The first launch status (wwrErrorInvalidValue if @p from has no
///         records, @p n_groups is not positive, or @p la is neither of
///         @p from's momenta).
[[nodiscard]] wwr::wwrError_t ordered_pair_order(wwr::wwrStream_t stream,
                                                 const PrimitivePairClassView &from, int32_t la,
                                                 int32_t n_groups, double *key, double *key_alt,
                                                 uint32_t *perm, uint32_t *perm_alt,
                                                 int32_t *group_start);

/// @brief Write the ordered class (@p la, lb) of the mmd class @p from in the
/// order @p perm gives (@ref ordered_pair_order's), on @p stream: record k is
/// slot perm[k], for k below @p n_rec, one thread per record.
///
/// As stored a record is mmd's with its bound_ab and rank_b; swapped, with
/// P - A and P - B exchanged, off_a and off_b exchanged, bound_ba, and rank_a
/// as its rank_b. @p doubles is the class's zeta, p_x, p_y, p_z, pa_x, pa_y,
/// pa_z, pb_x, pb_y, pb_z, s and record_bound, @p ints its off_a, off_b and
/// rank_b, each @p n_rec long and written whole; none may overlap another, or
/// @p from's fields.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive or @p la is neither of @p from's momenta).
[[nodiscard]] wwr::wwrError_t ordered_pair_gather(wwr::wwrStream_t stream,
                                                  const PrimitivePairClassView &from, int32_t la,
                                                  int32_t n_rec, const uint32_t *perm,
                                                  double *const doubles[12],
                                                  int32_t *const ints[3]);

/// @brief One bra class's rows of the (lb, ld) Dcd, `dc[r] = rank_b[r] * n_d`,
/// on @p stream, for its @p n_rec records' second-shell ranks @p rank_b and
/// @p n_d the number of shells of momentum ld: kb's `dc`, the row of Dcd
/// (exchange_gather_dcd) its record (a, c) reads, `rank_c * n_d + rank_d`
/// with the ket's `dd = rank_d`. It depends on the ket's ld, so a bra class
/// has one per ld; @p dc is @p n_rec int32s, written whole.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive or @p n_d is negative).
[[nodiscard]] wwr::wwrError_t ordered_pair_bra_rows(wwr::wwrStream_t stream, int32_t n_rec,
                                                    const int32_t *rank_b, int32_t n_d,
                                                    int32_t *dc);

} // namespace free_breakfast::device
