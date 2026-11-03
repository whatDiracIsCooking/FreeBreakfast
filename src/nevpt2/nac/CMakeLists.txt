# The module: the QD-NEVPT2 PT2 transition density, SC and PC, over the
# effective Hamiltonians' amplitudes. DeviceResources, MoBlocks, EnergyInputs,
# Buffers and Result are in its signatures, so core::resources, core::wwr,
# nevpt2::density_fitting, nevpt2::energy and core::error_handling are PUBLIC.
# The einsum and the FBF_TRY header are the implementation unit's alone. No
# device code of its own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.nac
  PRIMARY_INTERFACE
  nac.cppm
  IMPLEMENTATION
  pt2_density.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::energy
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::einsum
)
