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

/// @brief Fill @p ones[k] = 1 and @p rows[k] = k for k < @p n, on @p stream.
///
/// The inputs of the generated kernels a build without screening has no values
/// for: e_hermite's Schwarz bounds Ip and Iq (any finite, non-negative value
/// leaves `Dmax * Ip * Iq < tau` false at tau = 0) and jab's record-to-row map
/// of J_p, the identity with one batch per class.
///
/// @return The launch status (wwrErrorInvalidValue if @p n is not positive).
[[nodiscard]] wwr::wwrError_t coulomb_constants(wwr::wwrStream_t stream, int32_t n, double *ones,
                                                int32_t *rows);

/// @brief The pre-contraction of one ket class into @p dq and @p dmax, on
/// @p stream: each record's weight w (2 for a shell pair of two shells, 1 for a
/// shell against itself, read off its offsets) into @p w, then the generated
/// dq<la, lb> over the AO @p density.
///
/// @p density is n_ao x n_ao and symmetric (it is read at one orientation of
/// each block); @p dq is n_rec x n_hermite(la + lb) and @p dmax n_rec, both
/// written whole. cls.n_rec must be positive, cls.la <= cls.lb <=
/// kMaxAngularMomentum, and n_ao * n_ao must fit an int32_t.
///
/// @return The first launch status (wwrErrorInvalidValue for a class with no
///         kernel). Launch errors only, as overlap_class.
[[nodiscard]] wwr::wwrError_t coulomb_dq(wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                         int32_t n_ao, const double *density, double *w, double *dq,
                                         double *dmax);

/// @brief Add the [p|q] contraction of bra class @p bra against ket class
/// @p ket into the bra's @p jp, on @p stream: the generated
/// e_hermite<bra.la + bra.lb, ket.la + ket.lb>, one warp per bra record.
///
/// @p dq and @p dmax are @p ket's, from @ref coulomb_dq; @p jp is
/// bra.n_rec x n_hermite(bra.la + bra.lb), added to atomically, so it must be
/// zeroed before the first ket class and then holds the sum over all of them.
/// @p ones holds at least max(bra.n_rec, ket.n_rec) ones (@ref
/// coulomb_constants). Unscreened: tau = 0, so the kets' order is free.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t coulomb_e_hermite(wwr::wwrStream_t stream,
                                                const PrimitivePairClassView &bra,
                                                const PrimitivePairClassView &ket,
                                                const double *ones, const double *dq,
                                                const double *dmax, double *jp);

/// @brief Add one bra class's back-transform of @p jp into the n_ao x n_ao
/// matrix @p j, on @p stream: the generated jab<la, lb>, one thread per record.
///
/// @p jp is complete (every ket class added, @ref coulomb_e_hermite); @p rows
/// holds at least cls.n_rec entries, rows[k] = k (@ref coulomb_constants). Each
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

} // namespace free_breakfast::device
