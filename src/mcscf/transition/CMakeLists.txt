# The module: mcscf::transition_orbital_gradient, host code over
# response::fock and one axpy, with no kernels of its own. DeviceResources,
# Molecule, SphericalMatrixAO, Buffers and Result are in its signature, and
# MemoryFootprint in its footprint's, so core::resources, core::wwr, molecule,
# recipes::matrix_ao, core::error_handling and core::footprint are PUBLIC.
# The calls it makes are PRIVATE. It imports no other mcscf module, so
# gradient and nac can both take it.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.transition
  PRIMARY_INTERFACE
  transition.cppm
  IMPLEMENTATION
  transition.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
  free_breakfast::core::footprint
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::response::fock
)
