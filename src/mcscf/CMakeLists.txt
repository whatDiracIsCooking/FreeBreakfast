# CASSCF over integrals, ci and core, never pipeline: the energy, Fock
# matrices and orbital gradient at fixed orbitals and RDMs, the orbital Hessian
# over them, and the macro loop that optimizes both.
add_subdirectory(fock)
add_subdirectory(hessian)
add_subdirectory(solve)

# The aggregate: one import for the MCSCF layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf
  PRIMARY_INTERFACE
  mcscf.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::mcscf::fock
  free_breakfast::mcscf::hessian
  free_breakfast::mcscf::solve
)
