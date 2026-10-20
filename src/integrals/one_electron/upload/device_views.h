/// @file
/// @brief Non-owning views of the one-electron device inputs, for the host
/// module and the device launchers alike.
///
/// This header is the single definition of NucleiView, and the launchers' one
/// include for both views: PrimitivePairClassView is mmd's
/// (primitive_pair_class_view.h). The launchers (`*_launch.cu`, through their
/// bridge headers) reach it by `#include <device_views.h>`; C++ modules reach
/// the very same entities by `import free_breakfast.integrals.one_electron.upload;`,
/// which includes this header in its global module fragment and re-exports the
/// names from within the namespace.
///
/// The owners, @c PrimitivePairClass and @c DeviceNuclei, hold MirroredBuffers and
/// so live only in modules; each hands out one of these with @c view(). A view
/// is valid while its owner is alive.

#pragma once

#include "integral_types.h"
#include "primitive_pair_class_view.h"

namespace free_breakfast {

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
