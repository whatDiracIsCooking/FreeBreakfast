# The module: mcscf::fock and the packed order's two copies, declared in the
# interface and defined in the implementation unit. Molecule, BasisSet,
# DeviceResources, Buffers, the two kinds of pair class and the AO matrices
# are in its signature, so molecule, basis, core::wwr, core::resources,
# coulomb, exchange (which re-export their classes) and recipes::matrix_ao
# are PUBLIC. It launches nothing of its own: J and K are coulomb's and
# exchange's, the rest BLAS and copies, so there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.fock
  PRIMARY_INTERFACE
  fock.cppm
  IMPLEMENTATION
  fock.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
)
