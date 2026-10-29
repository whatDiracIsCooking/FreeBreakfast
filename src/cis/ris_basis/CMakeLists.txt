# The RIS minimal auxiliary basis: host-only, no device library. Its interface
# takes Atoms and returns a Result<BasisSet>, so atom, basis and
# core::error_handling are PUBLIC. The implementation unit alone reads Z
# (molecule's nuclear_charge) and converts the radii to bohr (core::constants).
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.cis.ris_basis
  PRIMARY_INTERFACE
  ris_basis.cppm
  IMPLEMENTATION
  ris_basis.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::atom
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::molecule
)
