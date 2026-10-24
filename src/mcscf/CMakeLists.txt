# CASSCF over integrals and core, never pipeline: the energy, Fock matrices and
# orbital gradient at fixed orbitals and RDMs.
add_subdirectory(fock)

# The aggregate: one import for the MCSCF layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf
  PRIMARY_INTERFACE
  mcscf.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::mcscf::fock
)
