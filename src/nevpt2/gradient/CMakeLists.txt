# The module: nevpt2::sc_energy and nevpt2::sc_adjoint. ci::strings,
# nevpt2::energy (ClassEnergies) and core::wwr (Buffers) are in its
# signatures, so they are PUBLIC with core::resources and
# core::error_handling. The builders' einsums, the BLAS calls, b_mo, rdms and
# the density-fitted adjoints are the implementation unit's alone. No device
# code of its own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  gradient.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::ci::strings
  free_breakfast::nevpt2::energy
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::einsum
  free_breakfast::recipes::symmetrize
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::rdm
)
