/// @file
/// @brief NucleiView: a molecule's nuclei on the device, by pointer, for the
/// host module and the device launchers alike.
///
/// This header is NucleiView's single definition. The launchers (`*_launch.cu`,
/// through their bridge headers) reach it by `#include <nuclei_view.h>`; C++
/// modules reach the very same entity by `import free_breakfast.molecule;`,
/// which includes this header in its global module fragment and re-exports the
/// name from within the namespace.
///
/// The owner, @c Molecule, holds MirroredBuffers and so lives only in a module;
/// it hands out a NucleiView with @c view(). A view is valid while its owner is
/// alive.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief A molecule's nuclei on the device: the arrays of a @c Molecule, by
/// pointer.
struct NucleiView {
  /// @brief The number of nuclei, positive.
  int32_t n_nuc;
  /// @brief Z_C, each nucleus's charge (n_nuc).
  const double *z;
  /// @brief C, each nucleus's position in bohr (3 per nucleus).
  const double *c;
};

} // namespace free_breakfast
