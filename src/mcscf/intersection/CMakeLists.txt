# The module: mcscf::df_intersection, host code over the CI objects, the
# generalized Fock, the orbital Hessian, response's relaxed_hessian and
# zvector modules (linked PRIVATE), the transition orbital gradient and
# the gradient module's calls, with no kernels of its own. DeviceResources,
# Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO, Buffers, Solution,
# Result and MemoryFootprint are in its signature, so core::resources,
# core::wwr, molecule, basis, mmd, recipes::matrix_ao, mcscf::solve,
# core::error_handling and core::footprint are PUBLIC. The calls it makes are
# PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.intersection
  PRIMARY_INTERFACE
  intersection.cppm
  IMPLEMENTATION
  intersection.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::integrals::mmd
  free_breakfast::recipes::matrix_ao
  free_breakfast::mcscf::solve
  free_breakfast::core::footprint
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::core::primitives
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::hessian
  free_breakfast::response::relaxed_hessian
  free_breakfast::response::zvector
  free_breakfast::mcscf::gradient
  free_breakfast::mcscf::transition
)
