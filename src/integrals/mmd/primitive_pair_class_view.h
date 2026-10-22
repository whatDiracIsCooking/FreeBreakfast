/// @file
/// @brief PrimitivePairClassView: one primitive-pair class's records on the
/// device, by pointer, for the launchers.
///
/// This header is the single definition. The launchers (`*_launch.cu`, through
/// their bridge headers) reach it by `#include`; C++ modules
/// reach the very same entity by `import free_breakfast.integrals.mmd;`, which
/// includes this header in its global module fragment and re-exports the name
/// (the `atom` pattern).
///
/// The owner, @c PrimitivePairClass, holds wwr buffers and so lives only in the
/// module; it hands out one of these with @c view(). A view is valid while its
/// owner is alive.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief One primitive-pair class on the device: the record arrays of a
/// @c PrimitivePairClass, by pointer.
///
/// Every array has @ref n_rec entries; a vector is three arrays, one per axis,
/// as the generated kernels take it.
struct PrimitivePairClassView {
  /// @brief The bra shell's angular momentum, <= @ref lb in a class's own
  /// view; a swapped view, which only ordered_pair_bound reads
  /// (mmd_bridge.h), may have it above.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive.
  int32_t n_rec;
  /// @brief zeta = alpha + beta.
  const double *zeta;
  /// @brief P: x.
  const double *p_x;
  /// @brief P: y.
  const double *p_y;
  /// @brief P: z.
  const double *p_z;
  /// @brief P - A: x.
  const double *pa_x;
  /// @brief P - A: y.
  const double *pa_y;
  /// @brief P - A: z.
  const double *pa_z;
  /// @brief P - B: x.
  const double *pb_x;
  /// @brief P - B: y.
  const double *pb_y;
  /// @brief P - B: z.
  const double *pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  const double *s;
  /// @brief alpha, the bra primitive's exponent.
  const double *ea;
  /// @brief beta, the ket primitive's exponent.
  const double *eb;
  /// @brief The record's Schwarz bound I = pair_bound(zeta, la + lb): a J or
  /// K screen's Ip as a bra and Iq as a ket.
  const double *bound;
  /// @brief The record's dressed Schwarz bound as stored: ao.k.record_bounds,
  /// the max over the bra shell's Cartesians.
  const double *bound_ab;
  /// @brief The record's dressed Schwarz bound swapped: the max over the ket
  /// shell's Cartesians.
  const double *bound_ba;
  /// @brief The bra shell's first Cartesian AO.
  const int32_t *off_a;
  /// @brief The ket shell's first Cartesian AO.
  const int32_t *off_b;
  /// @brief The bra shell's rank: its index among the basis's shells of
  /// momentum @ref la, in basis order.
  const int32_t *rank_a;
  /// @brief The ket shell's rank, among the shells of momentum @ref lb.
  const int32_t *rank_b;
};

} // namespace free_breakfast
