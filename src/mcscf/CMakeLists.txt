# CASSCF over response, integrals, ci and core, never pipeline: the macro loop
# that optimizes the orbitals and the CI over response's energy, gradient and
# orbital Hessian.
#
# relaxed_hessian first: solve links it. It is internal to the layer, so the
# aggregate below does not re-export it.
add_subdirectory(relaxed_hessian)
add_subdirectory(solve)
# zvector over relaxed_hessian, internal too: the SA gradient links it.
add_subdirectory(zvector)
# The DF-CASSCF nuclear gradient at a solve's solution, state-specific or one
# root of a state average (over zvector's multipliers).
add_subdirectory(gradient)
# F^[x] and the CP-CASSCF right-hand sides, the DF-CASSCF Hessian's
# per-coordinate derivatives at fixed parameters.
add_subdirectory(derivative)
# The equal-weight SA Hessian's root-rotation term and the roots' rotations
# theta^x, over derivative's H~^x and the orbital part of lambda^x.
add_subdirectory(root_rotation)
# The third derivative of the SA energy in its parameters along a fixed z: a
# state-averaged (or NEVPT2) nuclear Hessian's z . d2 g_SA term.
add_subdirectory(third_derivative)

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
  free_breakfast::mcscf::derivative
  free_breakfast::mcscf::root_rotation
  free_breakfast::mcscf::third_derivative
)
