/// @file
/// @brief The J build's launchers, declared once for the module that calls them
/// and the device TU that defines them.
///
/// Included by coulomb.cpp in its GLOBAL MODULE FRAGMENT, and by
/// coulomb_launch.cu directly, for the reason overlap_bridge.h gives: a
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

/// @brief Fill @p rows[k] = k for k < @p n, on @p stream: jab's record-to-row
/// map of J_p, the identity with one batch per class.
///
/// @return The launch status (wwrErrorInvalidValue if @p n is not positive).
[[nodiscard]] wwr::wwrError_t coulomb_rows(wwr::wwrStream_t stream, int32_t n, int32_t *rows);

/// @brief One class's inputs to every later step, on @p stream: each record's
/// weight w (2 for a shell pair of two shells, 1 for a shell against itself,
/// read off its offsets) into @p w, then the pre-contraction of the AO
/// @p density into @p dq and @p dmax (the generated dq<la, lb>). The record
/// bound, e_hermite's Ip and Iq, is the class's own (cls.bound, mmd's).
///
/// @p density is n_ao x n_ao and symmetric (it is read at one orientation of
/// each block); @p dq is q-major, n_hermite(la + lb) x n_rec (so e_hermite's
/// lanes, consecutive kets, read it contiguously), and @p w and @p dmax
/// n_rec, all written whole. cls.n_rec must be positive, cls.la <=
/// cls.lb <= kMaxAngularMomentum, and n_ao * n_ao must fit an int32_t.
///
/// @return The first launch status (wwrErrorInvalidValue for a class with no
///         kernel). Launch errors only, as overlap_class.
[[nodiscard]] wwr::wwrError_t coulomb_prepare(wwr::wwrStream_t stream,
                                              const PrimitivePairClassView &cls, int32_t n_ao,
                                              const double *density, double *w, double *dq,
                                              double *dmax);

/// @brief Add the [p|q] contraction of bra class @p bra against ket class
/// @p ket into the bra's @p jp, on @p stream: the generated
/// e_hermite<bra.la + bra.lb, ket.la + ket.lb>, one warp per bra record.
///
/// @p ip is @p bra's bound (bra.bound) and @p iq, @p dq and @p dmax are
/// @p ket's, the last two from @ref coulomb_prepare, @p dq q-major; @p jp is bra.n_rec x
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
[[nodiscard]] wwr::wwrError_t coulomb_e_hermite(wwr::wwrStream_t stream,
                                                const PrimitivePairClassView &bra,
                                                const PrimitivePairClassView &ket, double tau,
                                                const double *ip, const double *iq,
                                                const double *dq, const double *dmax, double *jp);

/// @brief Add one bra class's back-transform of @p jp into the n_ao x n_ao
/// matrix @p j, on @p stream: the generated jab<la, lb>, one thread per record.
///
/// @p jp is complete (every ket class added, @ref coulomb_e_hermite); @p rows
/// holds at least cls.n_rec entries, rows[k] = k (@ref coulomb_rows). Each
/// value is atomically added at (m, n) and (n, m), a shell against itself
/// writing its block's lower triangle and mirror once, so @p j must be zeroed
/// before the first class and is symmetric after the last. The kernels index
/// @p j as a 32-bit `m * n_ao + n`.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t coulomb_jab(wwr::wwrStream_t stream,
                                          const PrimitivePairClassView &cls, int32_t n_ao,
                                          const int32_t *rows, const double *jp, double *j);

/// @brief Raise @p maxima[0] to the largest dressed bound B (cls.bound_ab) of
/// @p cls's records and @p maxima[1] to the largest K = w ncart(la) B, on
/// @p stream: coulomb_pair_classes' maxima, one class at a time.
///
/// w is 1 for a record whose offsets are equal (a shell against itself) and 2
/// otherwise, as @ref coulomb_prepare's. @p maxima is two doubles, zero or
/// non-negative on entry, raised atomically (as the bits of a non-negative
/// double), so any number of classes may fold into it. Only la, n_rec,
/// bound_ab, off_a and off_b are read.
///
/// @return The launch status (wwrErrorInvalidValue if cls.n_rec is not
///         positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t
coulomb_prune_maxima(wwr::wwrStream_t stream, const PrimitivePairClassView &cls, double *maxima);

/// @brief Each record of @p cls's place among the class's kept records, on
/// @p stream: @p dest[r] is the number of kept records before r if r is kept,
/// -1 if not, and @p count[0] the number kept.
///
/// Record r is kept when `B_r Dc maxK >= tau || K_r Dc maxB >= tau`, B, K and
/// the maxima as in @ref coulomb_prune_maxima (@p maxima its result, complete
/// over every class) and Dc = 2, the density cap coulomb_pair_classes
/// assumes; @p tau = 0 keeps every record. The scan is stable, so the kept
/// records keep their order. @p dest is cls.n_rec long, written whole. Only
/// la, n_rec, bound_ab, off_a and off_b are read.
///
/// @return The launch status (wwrErrorInvalidValue if cls.n_rec is not
///         positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t coulomb_prune_keep(wwr::wwrStream_t stream,
                                                 const PrimitivePairClassView &cls,
                                                 const double *maxima, double tau, int32_t *dest,
                                                 int32_t *count);

/// @brief Scatter each kept record of @p from into @p to, on @p stream:
/// to[dest[r]] = from[r] for every r < @p n_rec with dest[r] >= 0
/// (@ref coulomb_prune_keep's @p dest).
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive). Launch errors only.
[[nodiscard]] wwr::wwrError_t coulomb_prune_scatter(wwr::wwrStream_t stream, int32_t n_rec,
                                                    const int32_t *dest, const double *from,
                                                    double *to);

/// @copydoc coulomb_prune_scatter
[[nodiscard]] wwr::wwrError_t coulomb_prune_scatter(wwr::wwrStream_t stream, int32_t n_rec,
                                                    const int32_t *dest, const int32_t *from,
                                                    int32_t *to);

} // namespace free_breakfast::device
