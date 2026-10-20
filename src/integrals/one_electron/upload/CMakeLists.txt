# The per-class launch and the shared pre-launch checks. No device code of its
# own: it validates, and every family's module launches. DeviceResources,
# DeviceStream, ForkStream, PrimitivePairClass, Molecule and CartesianMatrixAO
# are in its signatures, so all of their modules are PUBLIC; basis for
# kMaxAngularMomentum, the limit the check holds every class to.
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
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::linalg::matrix_ao
)
