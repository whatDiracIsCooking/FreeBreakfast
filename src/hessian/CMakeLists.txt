# The module: host code over the derivative and second-derivative integrals,
# the density-fitted J and K, the CPHF and wwr's BLAS, with no kernels of its
# own. DeviceResources,
# Buffers, Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO, Result,
# response::Cphf, mcscf::Solution and mcscf::ZVectors are in its signatures,
# so core::resources, core::wwr, molecule, basis, mmd, recipes::matrix_ao,
# core::error_handling, response::cphf, mcscf::solve and mcscf::zvector are
# PUBLIC. The derivative calls, the J and K builds, the second-derivative
# integrals, the MO-basis fit, the dipole integrals and their derivative, the
# CASSCF's integrals, CI, Fock, orbital and relaxed Hessians, derivatives,
# root rotation and third derivative, the gemms and the FBF_TRY header are the
# implementation units' own, so PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  PARTITIONS
  atom_block.cppm
  IMPLEMENTATION
  df_casscf.cpp
  df_casscf_exy.cpp
  df_casscf_two_electron.cpp
  df_rhf.cpp
  df_sacasscf.cpp
  df_sacasscf_z.cpp
  df_rhf_response.cpp
  df_rhf_two_electron.cpp
  infrared.cpp
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
  free_breakfast::response::cphf
  free_breakfast::mcscf::solve
  free_breakfast::mcscf::zvector
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::one_electron::hessian
  free_breakfast::integrals::one_electron::gradient
  free_breakfast::integrals::one_electron::dipole
  free_breakfast::integrals::one_electron::dipole_gradient
  free_breakfast::integrals::eri::density_fitting::gradient
  free_breakfast::integrals::eri::density_fitting::jk
  free_breakfast::integrals::eri::density_fitting::hessian
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::hessian
  free_breakfast::response::orbital_pairs
  free_breakfast::mcscf::derivative
  free_breakfast::mcscf::relaxed_hessian
  free_breakfast::mcscf::root_rotation
  free_breakfast::mcscf::third_derivative
)
