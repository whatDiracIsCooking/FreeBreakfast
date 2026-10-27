/// @file
/// @brief The DF three-center integrals' limits, record layouts and launchers
/// over the generated eri3<la, lb, lP>, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by three_center.cpp in its GLOBAL MODULE FRAGMENT, and by
/// three_center_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there. The per-class launch table
/// itself is kernels/eri3.h's `eri3_kernels`, which only a device TU can
/// include; three_center_launch.cu checks it against these limits.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "angular_momentum.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief (mn|P)'s orbital L limit: the largest momentum of an orbital shell m
/// or n in any class it takes, f. Its kernels are emitted through it
/// (tools/emit/df_kernels.py's MAX_L, which the tool checks against this).
inline constexpr int32_t kThreeCenterMaxMomentum = 3;

/// @brief (mn|P)'s aux L limit: the largest momentum of an auxiliary shell P
/// in any class it takes, i (tools/emit/df_kernels.py's AUX_MAX_L, as
/// kMetricMaxMomentum is). It is above the orbital limit on purpose, and is
/// not checked against kMaxAngularMomentum.
inline constexpr int32_t kThreeCenterAuxMaxMomentum = 6;

static_assert(kThreeCenterMaxMomentum == kMaxAngularMomentum,
              "(mn|P)'s orbital limit and the tree-wide kMaxAngularMomentum "
              "disagree");

/// @brief The fields of one bra record, each a segment of n_bra_rec doubles, in
/// this order: zeta, P's x, y and z, P - A's, P - B's, and s = c_a c_b
/// (pi / zeta)^{3/2}: mmd's record fields of the same names.
inline constexpr int32_t kThreeCenterBraFields = 11;

/// @brief The fields of one aux record, each a segment of n_aux_rec doubles, in
/// this order: the centre's x, y and z, the exponent alpha, and the weight
/// s = c (pi / alpha)^{3/2}: the metric's aux record.
inline constexpr int32_t kThreeCenterAuxFields = 5;

/// @brief The fields of one cell (an AO shell pair crossed with an aux shell),
/// each a segment of n_cell int32s, in this order: the pair's bra record range
/// [ab_rec0, ab_rec1), the aux shell's record range [p_rec0, p_rec1), and the
/// first Cartesian AO of the pair's two shells and of the aux shell, off_a,
/// off_b and off_p.
inline constexpr int32_t kThreeCenterCellFields = 7;

/// @brief Write class (la, lb, lp) of the Cartesian (mn|P) into the n_ao x
/// n_ao x n_aux @p eri3, element (m, n, P) at (m n_ao + n) n_aux + P, on
/// @p stream: the generated eri3<la, lb, lp>, one thread per cell.
///
/// @p bra is kThreeCenterBraFields segments of @p n_bra_rec doubles, @p aux
/// kThreeCenterAuxFields segments of @p n_aux_rec, and @p cells
/// kThreeCenterCellFields segments of @p n_cell int32s, each cell's pair of
/// momenta (la, lb) and its aux shell of lp. Each element is a plain store,
/// with its (m, n) mirror: every element belongs to exactly one cell, so
/// classes launched into one @p eri3 need no ordering among them and no zeroing
/// before the first, and a run is bit-reproducible. A diagonal pair
/// (off_a == off_b) writes its whole block.
///
/// The kernels index @p eri3 as a 32-bit int, so n_ao^2 n_aux must fit one.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel,
///         not 0 <= la <= lb <= kThreeCenterMaxMomentum and
///         0 <= lp <= kThreeCenterAuxMaxMomentum, or a non-positive count).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t three_center_class(wwr::wwrStream_t stream, int32_t la, int32_t lb,
                                                 int32_t lp, int32_t n_bra_rec, const double *bra,
                                                 int32_t n_aux_rec, const double *aux,
                                                 int32_t n_cell, const int32_t *cells, int32_t n_ao,
                                                 int32_t n_aux, double *eri3);

/// @brief @p out = half of @p in plus half of its (m, n) transpose, on
/// @p stream: element (P, n, m) at P + n_aux (n + n_ao m) of each, so out is
/// exactly symmetric in m and n. @p in and @p out must not overlap.
///
/// n_ao^2 n_aux must fit an int32_t.
///
/// @return The launch status (wwrErrorInvalidValue for a non-positive count).
[[nodiscard]] wwr::wwrError_t three_center_symmetrize(wwr::wwrStream_t stream, int32_t n_ao,
                                                      int32_t n_aux, const double *in, double *out);

} // namespace free_breakfast::device
