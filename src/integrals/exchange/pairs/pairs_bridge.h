/// @file
/// @brief The ordered classes' record-bound and record-order launchers (the
/// bound also over mmd's own records, for J's pruning),
/// declared once for the module that calls them and the device TU that
/// defines them.
///
/// Included by pairs.cpp in its GLOBAL MODULE FRAGMENT, and by
/// pairs_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.
///
/// Every pointer is device memory, every launch is asynchronous on @p stream,
/// and the launchers return launch errors only.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "ordered_pair_class_view.h"
#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Each record's DRESSED Schwarz bound into @p bound, on @p stream: the
/// generated record_bound<cls.la, cls.lb> (kernels/record_bound/), one thread
/// per record, FelizJueves's ao.k.record_bounds,
///
///     I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)
///
/// from the class's zeta, P - A, P - B and s, "a" its records' first shell.
/// The class is ORDERED: the max runs over the first shell's Cartesians, so
/// (la, lb) and (lb, la) are different kernels with different values. @p bound
/// is cls.n_rec long, written whole, and must not overlap the class's fields;
/// a pair whose pair factor underflows gets exactly 0.0.
///
/// @return The launch status (wwrErrorInvalidValue if the class has no
///         records or no kernel: la and lb at most kMaxAngularMomentum).
[[nodiscard]] wwr::wwrError_t ordered_pair_bound(wwr::wwrStream_t stream,
                                                 const OrderedPairClassView &cls, double *bound);

/// @brief The same bound of each of mmd's records, in its stored orientation:
/// "a" its bra shell, of the lower momentum, so record_bound<cls.la, cls.lb>.
/// J's pruning reads it (coulomb_pair_classes): a record's max over its bra
/// Cartesians, and so, times ncart(la), its sum over all of them. Otherwise as
/// above.
[[nodiscard]] wwr::wwrError_t ordered_pair_bound(wwr::wwrStream_t stream,
                                                 const PrimitivePairClassView &cls, double *bound);

/// @brief The order that puts each of a class's groups in descending @p bound,
/// on @p stream: record perm[k] is the k-th, the groups (@p group_start,
/// @p n_groups + 1 entries, the last @p n_rec) kept where they are, and equal
/// bounds within a group in record order.
///
/// Two stable sorts (src/integrals/sort's merge_sort_passes): every record by
/// -bound, then the result by its group's index, which leaves each group's
/// records in the first sort's order. @p key, @p key_alt, @p perm and
/// @p perm_alt are @p n_rec long; the alts are touched only past one tile of
/// the sort (2816 records). No array may overlap another.
///
/// @return The first launch status (wwrErrorInvalidValue if @p n_rec or
///         @p n_groups is not positive).
[[nodiscard]] wwr::wwrError_t ordered_pair_order(wwr::wwrStream_t stream, int32_t n_rec,
                                                 int32_t n_groups, const int32_t *group_start,
                                                 const double *bound, double *key, double *key_alt,
                                                 uint32_t *perm, uint32_t *perm_alt);

/// @brief Put @p field in the order @p perm gives, in place, on @p stream:
/// field[k] becomes the old field[perm[k]].
///
/// Copies @p field into @p tmp, then gathers it back. All three are @p n_rec
/// long, @p perm a permutation of 0 .. n_rec - 1 (@ref ordered_pair_order's).
///
/// @return The first copy or launch status (wwrErrorInvalidValue if @p n_rec
///         is not positive).
[[nodiscard]] wwr::wwrError_t ordered_pair_permute(wwr::wwrStream_t stream, int32_t n_rec,
                                                   const uint32_t *perm, double *field,
                                                   double *tmp);

/// @copydoc ordered_pair_permute
[[nodiscard]] wwr::wwrError_t ordered_pair_permute(wwr::wwrStream_t stream, int32_t n_rec,
                                                   const uint32_t *perm, int32_t *field,
                                                   int32_t *tmp);

} // namespace free_breakfast::device
