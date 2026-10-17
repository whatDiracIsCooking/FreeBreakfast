# The module. No device code of its own: H is kinetic's T plus potential_en's
# add-only V, both launched by their own modules. DeviceResources,
# DevicePrimitivePairClass, DeviceNuclei and CartesianMatrixAO are in its
# signature, so core::resources, one_electron::upload and linalg::matrix_ao are
# PUBLIC; kinetic and potential_en are only called, so PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.core_hamiltonian
  PRIMARY_INTERFACE
  core_hamiltonian.cppm
  IMPLEMENTATION
  core_hamiltonian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::one_electron::upload
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::kinetic
  free_breakfast::integrals::one_electron::potential_en
)
