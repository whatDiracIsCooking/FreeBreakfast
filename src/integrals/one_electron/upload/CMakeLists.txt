# The non-owning device views, offered two ways from one header, device_views.h:
# `#include <device_views.h>` for the launchers, through this include path, and
# `import free_breakfast.integrals.one_electron.upload;`, which includes it in
# its GMF and re-exports the names. PrimitivePairClassView is mmd's
# (primitive_pair_class_view.h), which device_views.h includes.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.one_electron.upload.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
)

# The nuclei, the per-class launch and the shared pre-launch check. No device
# code of its own: it copies and validates, and every family's module launches.
# DeviceResources, MirroredBuffer, DeviceStream, ForkStream, PrimitivePairClass,
# Molecule and CartesianMatrixAO are in its signatures, so all of their modules
# are PUBLIC;
# basis for kMaxAngularMomentum, the limit the check holds every class to.
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
