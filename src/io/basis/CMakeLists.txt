# basis and molecule are PUBLIC: the reader's signatures name BasisStamp,
# BasisSet and Molecule.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.io.basis
  PRIMARY_INTERFACE
  basis.cppm
  PARTITIONS
  basis_reader.cppm
  IMPLEMENTATION
  basis_reader.cpp
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  free_breakfast::molecule
  LINK_PRIVATE
  free_breakfast::io::base
  free_breakfast::io::json
  free_breakfast::core::diagnostics_macros
  IMPORT_STD
)
