# The per-class launch and the shared pre-launch checks. No device code of its
# own: it validates, and every family's module launches. DeviceResources,
# DeviceStream, ForkStream, PrimitivePairClass, Molecule and CartesianMatrixAO
# are in its signatures, so all of their modules are PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.launch
  PRIMARY_INTERFACE
  launch.cppm
  IMPLEMENTATION
  launch.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
)
