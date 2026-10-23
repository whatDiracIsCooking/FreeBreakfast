/// @file
/// @brief The record, record-order and record-bound launchers, and the source they form the
/// records from, declared once for the module that calls them and the device TU that defines them.
///
/// Included by mmd.cpp in its GLOBAL MODULE FRAGMENT, and by mmd_launch.cu
/// directly, for the reason overlap_bridge.h gives: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU. Nothing above mmd includes it: coulomb and
/// exchange/pairs read the bounds these launchers write, not the launchers.
/// wwrStream_t and wwrError_t come from a different wwr header in each pass,
/// as there.

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

/// @brief What a class's records are formed from, on the device: the basis's
/// shells and primitives, and the class's shell pairs.
///
/// A record is named by its index in the order the host would have written it
/// (shell pair, then bra primitive, then ket primitive): the shell pair p whose
/// [rec_begin[p], rec_begin[p + 1]) holds it, then its offset in that range,
/// bra-primitive-major. One thread per record finds p by a binary search of
/// @ref rec_begin.
struct PrimitivePairSource {
  /// @brief The class's shell pairs, positive.
  int32_t n_pairs;
  /// @brief Each shell pair's bra shell, already in la <= lb order.
  const uint32_t *bra;
  /// @brief Each shell pair's ket shell.
  const uint32_t *ket;
  /// @brief Each shell pair's first record, n_pairs + 1 long: rec_begin[0] is
  /// 0, rec_begin[n_pairs] the class's record count.
  const int32_t *rec_begin;
  /// @brief Each shell's first primitive in @ref exponent and
  /// @ref coefficient, n_shells + 1 long.
  const int32_t *prim_begin;
  /// @brief Every primitive's exponent, shell by shell.
  const double *exponent;
  /// @brief Every primitive's contraction coefficient, as the basis gives it.
  const double *coefficient;
  /// @brief Each shell's centre: x.
  const double *c_x;
  /// @brief Each shell's centre: y.
  const double *c_y;
  /// @brief Each shell's centre: z.
  const double *c_z;
  /// @brief Each shell's first Cartesian AO.
  const int32_t *ao_offset;
  /// @brief Each shell's rank among the basis's shells of its momentum.
  const int32_t *rank;
};

/// @brief A class's record fields on the device, writable: what
/// @ref mmd_records fills. The bounds are not among them; mmd_bound and
/// ordered_pair_bound write those.
struct PrimitivePairRecords {
  /// @brief zeta = alpha + beta.
  double *zeta;
  /// @brief P: x.
  double *p_x;
  /// @brief P: y.
  double *p_y;
  /// @brief P: z.
  double *p_z;
  /// @brief P - A: x.
  double *pa_x;
  /// @brief P - A: y.
  double *pa_y;
  /// @brief P - A: z.
  double *pa_z;
  /// @brief P - B: x.
  double *pb_x;
  /// @brief P - B: y.
  double *pb_y;
  /// @brief P - B: z.
  double *pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  double *s;
  /// @brief alpha.
  double *ea;
  /// @brief beta.
  double *eb;
  /// @brief The bra shell's first Cartesian AO.
  int32_t *off_a;
  /// @brief The ket shell's first Cartesian AO.
  int32_t *off_b;
  /// @brief The bra shell's rank.
  int32_t *rank_a;
  /// @brief The ket shell's rank.
  int32_t *rank_b;
};

/// @brief The order of a class's records by zeta, descending, on @p stream:
/// the record of host index perm[k] is the k-th, equal zetas in host order.
///
/// Writes key[r] = -zeta of host record r (formed from @p src) and
/// perm[r] = r, then sorts the keys ascending with perm carried
/// (src/integrals/sort's merge_sort_passes, stable). @p key, @p key_alt,
/// @p perm and @p perm_alt are @p n_rec long; the alts are touched only past
/// one tile of the sort (2816 records). No array may overlap another.
///
/// @return The first launch status (wwrErrorInvalidValue if @p n_rec or
///         src.n_pairs is not positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t mmd_order(wwr::wwrStream_t stream, int32_t n_rec,
                                        const PrimitivePairSource &src, double *key,
                                        double *key_alt, uint32_t *perm, uint32_t *perm_alt);

/// @brief Every record field but the bounds, in the order @p perm gives, on
/// @p stream: record k is the record of host index perm[k], formed from
/// @p src, one thread per record.
///
/// P is (alpha A + beta B) / zeta with no FMA contraction, each product
/// rounded on its own: the IEEE value a host computes, bit for bit, and the
/// same for a record and its primitives swapped (the two records a diagonal
/// shell pair has for every pair of distinct primitives).
///
/// @p perm and every field of @p out are @p n_rec long, written whole.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec or
///         src.n_pairs is not positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t mmd_records(wwr::wwrStream_t stream, int32_t n_rec,
                                          const PrimitivePairSource &src, const uint32_t *perm,
                                          const PrimitivePairRecords &out);

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
/// P - B exchanged: mmd forms @ref PrimitivePairClass::bound_ba so. Only la,
/// lb, n_rec, zeta, pa_*, pb_* and s are read. @p bound is cls.n_rec long,
/// written whole, and must not overlap the view's fields; a pair whose pair
/// factor underflows gets exactly 0.0.
///
/// @return The launch status (wwrErrorInvalidValue if the view has no
///         records or no kernel: la and lb at most kMaxAngularMomentum).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t ordered_pair_bound(wwr::wwrStream_t stream,
                                                 const PrimitivePairClassView &cls, double *bound);

} // namespace free_breakfast::device
