/// @file
/// @brief Non-owning views of the one-electron device inputs, for the host
/// module and the device launchers alike.
///
/// This header is the single definition. The launchers (`*_launch.cu`, through
/// their bridge headers) reach it by `#include <device_views.h>`; C++ modules
/// reach the very same entities by `import
/// free_breakfast.integrals.one_electron.upload;`, which includes this header in
/// its global module fragment and re-exports the names from within the
/// namespace.
///
/// The owners, @c DevicePrimitivePairClass and @c DeviceNuclei, hold wwr
/// buffers and so live only in the module; each hands out one of these with
/// @c view(). A view is valid while its owner is alive and unmoved.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief One primitive-pair class on the device: the record arrays of a
/// @c DevicePrimitivePairClass, by pointer.
///
/// Every array has @ref n_rec entries; a vector is three arrays, one per axis,
/// as the generated kernels take it.
struct PrimitivePairClassView {
  /// @brief The bra shell's angular momentum, <= @ref lb.
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
  /// @brief beta, the ket primitive's exponent.
  const double *eb;
  /// @brief The bra shell's first Cartesian AO.
  const int32_t *off_a;
  /// @brief The ket shell's first Cartesian AO.
  const int32_t *off_b;
};

/// @brief A molecule's nuclei on the device: the arrays of a @c DeviceNuclei,
/// by pointer.
struct NucleiView {
  /// @brief The number of nuclei, positive.
  int32_t n_nuc;
  /// @brief Z_C, each nucleus's charge (n_nuc).
  const double *z;
  /// @brief C, each nucleus's position in bohr (3 per nucleus).
  const double *c;
};

} // namespace free_breakfast
