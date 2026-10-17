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

// PrimPairSoA, FelizJueves's record struct (deps/FelizJueves/include). At
// file scope, ahead of any launcher's namespaced #include of a generated
// kernel, whose own include of it is then a no-op (prim_pair.h's note).
#include "felizjueves/prim_pair.h"

namespace free_breakfast {

/// @brief One primitive-pair class on the device: the record arrays of a
/// @c DevicePrimitivePairClass, by pointer.
///
/// The arrays are @ref recs, a FelizJueves @c PrimPairSoA: one flat array of
/// @ref n_rec entries per field, every vector split per axis (XPA, YPA, ZPA).
/// It is the struct every generated one-electron kernel takes by value, and
/// the superset all of them read, so a launcher passes it through unchanged.
struct PrimitivePairClassView {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive.
  int32_t n_rec;
  /// @brief The record arrays: zeta, s, eb, P, P - A, P - B (per axis) and
  /// the two AO offsets (felizjueves/prim_pair.h has each field).
  PrimPairSoA recs;
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
