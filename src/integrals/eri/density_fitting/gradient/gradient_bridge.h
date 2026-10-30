/// @file
/// @brief The DF gradient kernels' record layouts and their per-class
/// launchers over the generated gmetric<lP, lQ> and geri3<la, lb, lP>,
/// declared once for the module that calls them and the device TU that
/// defines them.
///
/// Included by gradient.cpp in its GLOBAL MODULE FRAGMENT, and by
/// gradient_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. The limits are the energy kernels'
/// (kMetricMaxMomentum, kThreeCenterMaxMomentum, kThreeCenterAuxMaxMomentum),
/// from their bridges, and so is the aux record (kMetricRecordFields); the
/// launch tables are kernels/gmetric.h's and kernels/geri3.h's, which only a
/// device TU can include, and gradient_launch.cu checks them against those
/// limits.

#pragma once

// Standard (via integral_types.h, through both) before runtime.h: under HIP a
// libc++ header must not follow the vendor runtime header (see primitives.h).
#include "metric_bridge.h"
#include "three_center_bridge.h"

namespace free_breakfast::device {

/// @brief The fields of one aux shell pair (P, Q) of the metric gradient, each
/// a segment of n_pair int32s, in this order: the metric's six
/// (kMetricPairFields: P's and Q's record ranges, their first Cartesian AOs)
/// and then the atoms P and Q sit on, atom_p and atom_q.
inline constexpr int32_t kMetricGradientPairFields = kMetricPairFields + 2;

/// @brief The fields of one bra record of the three-center gradient, each a
/// segment of n_bra_rec doubles: the energy kernel's eleven
/// (kThreeCenterBraFields) and then the two primitives' exponents, alpha and
/// beta.
inline constexpr int32_t kThreeCenterGradientBraFields = kThreeCenterBraFields + 2;

/// @brief The fields of one cell of the three-center gradient, each a segment
/// of n_cell int32s: the energy kernel's seven (kThreeCenterCellFields) and
/// then the atoms the pair's two shells and the aux shell sit on, atom_a,
/// atom_b and atom_p.
inline constexpr int32_t kThreeCenterGradientCellFields = kThreeCenterCellFields + 3;

/// @brief Add class (lp, lq)'s share of SUM_PQ W_PQ d(P|Q)/dR into the
/// n_atom x 3 @p gradient, on @p stream: the generated gmetric<lp, lq>, one
/// thread per aux shell pair.
///
/// @p records is kMetricRecordFields segments of @p n_rec doubles (the
/// metric's aux records); @p pairs is kMetricGradientPairFields segments of
/// @p n_pair int32s, each pair's P of momentum @p lp and Q of @p lq, staged as
/// the metric stages them. @p w is the Cartesian n_aux x n_aux W, any matrix.
/// Each thread atomically adds into @p gradient, which the caller zeroes, so a
/// run is not bit-reproducible.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel,
///         not 0 <= lp <= lq <= kMetricMaxMomentum, or a non-positive count).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t metric_gradient_class(wwr::wwrStream_t stream, int32_t lp, int32_t lq,
                                                    int32_t n_rec, const double *records,
                                                    int32_t n_pair, const int32_t *pairs,
                                                    int32_t n_aux, int32_t n_atom, const double *w,
                                                    double *gradient);

/// @brief Add class (la, lb, lp)'s share of SUM_mnP Gam_mnP d(mn|P)/dR into
/// the n_atom x 3 @p gradient, on @p stream: the generated geri3<la, lb, lp>,
/// one thread per cell.
///
/// @p bra is kThreeCenterGradientBraFields segments of @p n_bra_rec doubles,
/// @p aux kMetricRecordFields segments of @p n_aux_rec, and @p cells
/// kThreeCenterGradientCellFields segments of @p n_cell int32s, staged as
/// three_center stages them. @p gamma is the Cartesian n_ao x n_ao x n_aux
/// Gam, element (m, n, P) at (m n_ao + n) n_aux + P, any tensor. Each thread
/// atomically adds into @p gradient, which the caller zeroes.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel,
///         not 0 <= la <= lb <= kThreeCenterMaxMomentum and
///         0 <= lp <= kThreeCenterAuxMaxMomentum, or a non-positive count).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t
three_center_gradient_class(wwr::wwrStream_t stream, int32_t la, int32_t lb, int32_t lp,
                            int32_t n_bra_rec, const double *bra, int32_t n_aux_rec,
                            const double *aux, int32_t n_cell, const int32_t *cells, int32_t n_ao,
                            int32_t n_aux, int32_t n_atom, const double *gamma, double *gradient);

} // namespace free_breakfast::device
