/// @file
/// @brief OrderedPairClassView: one ordered pair class's records on the
/// device, by pointer, for the exchange launchers.
///
/// This header is the single definition, as primitive_pair_class_view.h is
/// for mmd's classes: a launcher reaches it by `#include` (through its bridge
/// header), a module by `import free_breakfast.integrals.exchange.pairs;`,
/// which includes it in its global module fragment and re-exports the name.
///
/// The owner, @c OrderedPairClass, holds wwr buffers and so lives only in the
/// module; it hands out one of these with @c view(). A view is valid while its
/// owner is alive.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief One ordered pair class on the device: the record arrays of an
/// @c OrderedPairClass, by pointer.
///
/// Every array has @ref n_rec entries; a vector is three arrays, one per axis.
/// "a" is the record's first shell, the output index K is built on (kernel_k's
/// `a` on a bra, `b` on a ket), and "b" its second.
struct OrderedPairClassView {
  /// @brief The first shell's angular momentum.
  int32_t la;
  /// @brief The second shell's angular momentum.
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
  /// @brief P - A, A the first shell's centre: x.
  const double *pa_x;
  /// @brief P - A: y.
  const double *pa_y;
  /// @brief P - A: z.
  const double *pa_z;
  /// @brief P - B, B the second shell's centre: x.
  const double *pb_x;
  /// @brief P - B: y.
  const double *pb_y;
  /// @brief P - B: z.
  const double *pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  const double *s;
  /// @brief The record's Schwarz bound: kernel_k's Ip as a bra, Iq as a ket.
  const double *bound;
  /// @brief The first shell's first Cartesian AO.
  const int32_t *off_a;
  /// @brief The second shell's first Cartesian AO.
  const int32_t *off_b;
  /// @brief The second shell's rank: its index among the basis's shells of
  /// momentum @ref lb, in basis order (exchange_helpers_bridge.h's RANK).
  const int32_t *rank_b;
};

} // namespace free_breakfast
