# The module: mcscf::df_derivatives, host code over the derivative integrals,
# the density-fitted J and K, the fitted integrals, the generalized Fock, the
# CI sigma and RDMs and wwr's BLAS, with no kernels of its own. DeviceResources,
# Buffers, Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO and
# Result are in its signature, so core::resources, core::wwr, molecule, basis,
# mmd, recipes::matrix_ao and core::error_handling are PUBLIC. The calls it
# makes are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.derivative
  PRIMARY_INTERFACE
  derivative.cppm
  IMPLEMENTATION
  derivative.cpp
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
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::integrals::active_space
  free_breakfast::integrals::one_electron::gradient
  free_breakfast::integrals::eri::density_fitting::gradient
  free_breakfast::integrals::eri::density_fitting::jk
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::response::fock
)
