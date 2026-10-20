# The non-owning device views, offered two ways from one definition in
# device_views.h: `#include <device_views.h>` for the launchers, through this
# include path, and `import free_breakfast.integrals.one_electron.upload;`,
# which includes it in its GMF and re-exports the names.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.one_electron.upload.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# The device inputs and the shared pre-launch check. No device code of its own:
# it copies and validates, and every family's module launches. DeviceResources,
# Buffers, MirroredBuffer, BasisSet, the mmd record layout, Molecule and
# CartesianMatrixAO are in its signatures, so all of their modules are PUBLIC;
# basis also for kMaxAngularMomentum, the limit the check holds every class to.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.upload
  PRIMARY_INTERFACE
  upload.cppm
  IMPLEMENTATION
  upload.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::mirrored_buffer
  free_breakfast::core::resources
  free_breakfast::integrals::one_electron::upload::headers
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
