# CASSCF over response, integrals, ci and core, never pipeline: the macro loop
# that optimizes the orbitals and the CI over response's energy, gradient and
# orbital Hessian.
add_subdirectory(solve)

# The aggregate: one import for the MCSCF layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf
  PRIMARY_INTERFACE
  mcscf.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::mcscf::solve
)
