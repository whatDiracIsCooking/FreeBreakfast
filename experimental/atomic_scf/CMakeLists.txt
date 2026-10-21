# Host only: the integrals, the SCF and the assembly are plain C++ with no
# device code. basis is PUBLIC because the interface takes Shell, BasisSet and
# Atom (basis re-exports molecule, which re-exports atom); sad is PRIVATE, for
# the aufbau fill the implementation reuses.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.atomic_scf
  PRIMARY_INTERFACE
  atomic_scf.cppm
  IMPLEMENTATION
  atomic_scf.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::sad
)
