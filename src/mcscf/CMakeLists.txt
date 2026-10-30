# CASSCF over response, integrals, ci and core, never pipeline: the macro loop
# that optimizes the orbitals and the CI over response's energy, gradient and
# orbital Hessian.
#
# relaxed_hessian first: solve links it. It is internal to the layer, so the
# aggregate below does not re-export it.
add_subdirectory(relaxed_hessian)
add_subdirectory(solve)
# The state-specific DF-CASSCF nuclear gradient at a solve's solution.
add_subdirectory(gradient)

# The aggregate: one import for the MCSCF layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf
  PRIMARY_INTERFACE
  mcscf.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::mcscf::solve
  free_breakfast::mcscf::gradient
)
