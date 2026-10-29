# The module: cis::Sigma, cis::DfSigma, cis::Spin and cis::Options, from BLAS,
# the spherical bridge, nonsymmetric_jk and b_mo, so no device library. Its
# interface takes DeviceResources, a BasisSet, J's and K's classes, a
# SphericalMatrixAO and the fitted B's Buffers, holds MatrixAOPairs and
# Buffers, returns Status and carries DavidsonOptions, hence everything named
# there PUBLIC. The implementation units alone call the gemms,
# nonsymmetric_jk and b_mo, round the workspace, copy the scalars and check
# the LinearOperator concept, so those and the FBF_TRY header are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.cis.sigma
  PRIMARY_INTERFACE
  sigma.cppm
  IMPLEMENTATION
  sigma.cpp
  df_sigma.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::recipes::davidson
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::linear_operator
  free_breakfast::integrals::eri::integral_direct::nonsymmetric_jk
  free_breakfast::integrals::eri::density_fitting::mo
)
