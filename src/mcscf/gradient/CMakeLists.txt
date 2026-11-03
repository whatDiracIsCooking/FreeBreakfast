# The module: mcscf::df_gradient, host code over the fitted integrals, the
# generalized Fock, the shared one-electron terms, the density-fitting
# derivative contractions and wwr's BLAS, with no kernels of its own; for
# one root of a state average, the CI strings, sigma and RDMs, the orbital
# Hessian and the internal relaxed_hessian and zvector modules (neither is
# re-exported, and free_breakfast.mcscf does not re-export them either). And
# root_pair_multipliers and zeta_terms, over zvector's rotating_pairs,
# root_rotation's kRootGapFloor and transition's g^(sr).
# DeviceResources, Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO,
# Buffers, Solution, ZVector, ci::Roots, ci::Rdm and Result are in its
# signature, so core::resources, core::wwr, molecule, basis, mmd,
# recipes::matrix_ao, ci::solver, ci::rdm, mcscf::solve, mcscf::zvector and
# core::error_handling are PUBLIC. The calls it makes are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  gradient.cpp
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
  free_breakfast::ci::solver
  free_breakfast::ci::rdm
  free_breakfast::mcscf::solve
  free_breakfast::mcscf::zvector
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::active_space
  free_breakfast::integrals::eri::density_fitting::gradient
  free_breakfast::integrals::eri::density_fitting::jk
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::response::fock
  free_breakfast::response::hessian
  free_breakfast::response::orbital_pairs
  free_breakfast::gradient
  free_breakfast::mcscf::relaxed_hessian
  free_breakfast::mcscf::transition
  free_breakfast::mcscf::root_rotation
)
