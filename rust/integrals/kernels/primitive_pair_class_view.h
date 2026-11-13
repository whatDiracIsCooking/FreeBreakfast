/* fb_primitive_pair_class_view: one primitive-pair class's records on the
 * device, by pointer, for the launch wrappers. Forked from
 * src/integrals/mmd/primitive_pair_class_view.h, the same fields in the same
 * order, as plain C (rust/docs/gpu_layer.md, "Shared types"): mmd/mmd.h and
 * one_electron/overlap/overlap.h include it, and devtools/rust-bindings.py
 * reads it into fb-integrals-sys's #[repr(C)] mirror. */
#pragma once

#include <stdint.h>

/// One primitive-pair class on the device: the record arrays of a class, by
/// pointer. The struct itself is host memory, handed to a wrapper by address;
/// every array it points at is device memory with n_rec entries, and a vector
/// is three arrays, one per axis, as the generated kernels take it.
typedef struct fb_primitive_pair_class_view {
  /// The bra shell's angular momentum, at most lb in the view of a basis's
  /// own class; a mixed class, or a swapped view, which only
  /// fb_ordered_pair_bound and the mixed overlap read, may have it above.
  int32_t la;
  /// The ket shell's angular momentum.
  int32_t lb;
  /// The number of records, positive.
  int32_t n_rec;
  /// zeta = alpha + beta.
  const double *zeta;
  /// P: x.
  const double *p_x;
  /// P: y.
  const double *p_y;
  /// P: z.
  const double *p_z;
  /// P - A: x.
  const double *pa_x;
  /// P - A: y.
  const double *pa_y;
  /// P - A: z.
  const double *pa_z;
  /// P - B: x.
  const double *pb_x;
  /// P - B: y.
  const double *pb_y;
  /// P - B: z.
  const double *pb_z;
  /// c_a c_b (pi / zeta)^{3/2}.
  const double *s;
  /// alpha, the bra primitive's exponent.
  const double *ea;
  /// beta, the ket primitive's exponent.
  const double *eb;
  /// The record's Schwarz bound I = pair_bound(zeta, la + lb): a J or K
  /// screen's Ip as a bra and Iq as a ket.
  const double *bound;
  /// The record's dressed Schwarz bound as stored: the max over the bra
  /// shell's Cartesians.
  const double *bound_ab;
  /// The record's dressed Schwarz bound swapped: the max over the ket
  /// shell's Cartesians.
  const double *bound_ba;
  /// The bra shell's first Cartesian AO.
  const int32_t *off_a;
  /// The ket shell's first Cartesian AO.
  const int32_t *off_b;
  /// The bra shell's rank: its index among the basis's shells of momentum
  /// la, in basis order.
  const int32_t *rank_a;
  /// The ket shell's rank, among the shells of momentum lb.
  const int32_t *rank_b;
} fb_primitive_pair_class_view;
