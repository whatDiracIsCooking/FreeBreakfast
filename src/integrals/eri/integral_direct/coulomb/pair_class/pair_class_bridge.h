/// @file
/// @brief The prune's launchers, declared once for the module that calls them
/// and the device TU that defines them.
///
/// Included by pair_class.cpp in its GLOBAL MODULE FRAGMENT, and by
/// pair_class_launch.cu directly, for the reason overlap_bridge.h gives: a
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

/// @brief J's L limit: the largest momentum of a shell in any class the J
/// build takes, f. Its kernels are emitted through it
/// (tools/emit/coulomb_kernels.py's MAX_L), and dq_launch.cu,
/// e_hermite_launch.cu and j_coulomb_launch.cu check it against their tables,
/// which a host module unit cannot include; the tree-wide kMaxAngularMomentum
/// is not J's. Here, in the lowest of J's bridges, because
/// coulomb_pair_classes refuses a class above it as coulomb does.
inline constexpr int32_t kCoulombMaxMomentum = 3;

/// @brief Raise @p maxima[0] to the largest dressed bound B (cls.bound_ab) of
/// @p cls's records and @p maxima[1] to the largest K = w ncart(la) B, on
/// @p stream: coulomb_pair_classes' maxima, one class at a time.
///
/// w is 1 for a record whose offsets are equal (a shell against itself) and 2
/// otherwise, as dq's. @p maxima is two doubles, zero or non-negative on
/// entry, raised atomically (as the bits of a non-negative double), so any
/// number of classes may fold into it. Only la, n_rec, bound_ab, off_a and
/// off_b are read.
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

/// @brief Fill @p rows[k] = k for k < @p n_rec, on @p stream: a class's
/// CoulombPairClass::row, jab's record-to-row map of J_p, the identity with
/// one batch per class.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive).
[[nodiscard]] wwr::wwrError_t coulomb_rows(wwr::wwrStream_t stream, int32_t n_rec, int32_t *rows);

} // namespace free_breakfast::device
