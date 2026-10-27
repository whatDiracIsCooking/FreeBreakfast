# The derivatives of a CASSCF energy with respect to the orbital rotation, at
# fixed orbitals and RDMs, over integrals and core, never pipeline: the energy,
# Fock matrices and orbital gradient, and the orbital Hessian over them. What
# an optimizer steps over and what a response solve inverts.
add_subdirectory(fock)
add_subdirectory(hessian)

# The aggregate: one import for the response layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.response
  PRIMARY_INTERFACE
  response.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::response::fock
  free_breakfast::response::hessian
)
