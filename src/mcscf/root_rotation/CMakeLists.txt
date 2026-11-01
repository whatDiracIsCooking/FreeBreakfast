# The module: mcscf::root_rotation, host code over the fitted integrals, the
# Fock, the CI RDMs, relaxed_hessian's response_gradient and wwr's BLAS, with
# no kernels of its own. DeviceResources, Buffers, Molecule, BasisSet,
# SphericalMatrixAO, Derivatives and Result are in its signature, so those
# targets are PUBLIC. The calls it makes are PRIVATE, relaxed_hessian (internal
# to the layer) among them.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.root_rotation
  PRIMARY_INTERFACE
  root_rotation.cppm
  IMPLEMENTATION
  root_rotation.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
  free_breakfast::mcscf::derivative
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::orbital_pairs
  free_breakfast::mcscf::relaxed_hessian
)
