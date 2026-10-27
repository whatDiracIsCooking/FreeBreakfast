/// @file
/// @brief The DF Coulomb metric's limit and its per-class launcher over the
/// generated metric<lP, lQ>, declared once for the module that calls it and
/// the device TU that defines it.
///
/// Included by metric.cpp in its GLOBAL MODULE FRAGMENT, and by
/// metric_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there. The per-class launch table
/// itself is kernels/metric.h's `metric_kernels`, which only a device TU can
/// include; metric_launch.cu checks it against kMetricMaxMomentum.

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

/// @brief The metric's L limit: the largest momentum of an auxiliary shell in
/// any class (P|Q) it takes, g. Its kernels are emitted through it
/// (tools/emit/df_kernels.py's AUX_MAX_L, which the tool checks against this),
/// and metric_launch.cu checks their table against it; the tree-wide
/// kMaxAngularMomentum is the orbital basis's, not the aux basis's.
inline constexpr int32_t kMetricMaxMomentum = 4;

/// @brief The fields of one aux record, each a segment of n_rec doubles, in
/// this order: the centre's x, y and z, the exponent alpha, and the weight
/// s = c (pi / alpha)^{3/2}.
inline constexpr int32_t kMetricRecordFields = 5;

/// @brief The fields of one aux shell pair (P, Q), each a segment of n_pair
/// int32s, in this order: P's record range [p_rec0, p_rec1), Q's
/// [q_rec0, q_rec1), and the two shells' first Cartesian AOs off_p and off_q.
inline constexpr int32_t kMetricPairFields = 6;

/// @brief Write class (lp, lq) of the Cartesian metric (P|Q) into the
/// n_aux x n_aux @p metric, on @p stream: the generated metric<lp, lq>, one
/// thread per aux shell pair.
///
/// @p records is kMetricRecordFields segments of @p n_rec doubles, every record
/// the pairs name; @p pairs is kMetricPairFields segments of @p n_pair int32s,
/// each pair's P of momentum @p lp and Q of @p lq. Each element is a plain
/// store, with its mirror: every element belongs to exactly one aux shell
/// pair, so classes launched into one @p metric need no ordering among them
/// and no zeroing before the first, and a run is bit-reproducible. A diagonal
/// pair (P == Q, only when lp == lq) writes its whole block too.
///
/// The kernels index @p metric as a 32-bit `a * n_aux + b`, so n_aux^2 must
/// fit an int32_t.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel,
///         not 0 <= lp <= lq <= kMetricMaxMomentum, or a non-positive count).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t metric_class(wwr::wwrStream_t stream, int32_t lp, int32_t lq,
                                           int32_t n_rec, const double *records, int32_t n_pair,
                                           const int32_t *pairs, int32_t n_aux, double *metric);

} // namespace free_breakfast::device
