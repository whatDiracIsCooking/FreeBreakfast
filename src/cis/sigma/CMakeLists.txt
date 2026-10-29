# The module: cis::Sigma, cis::Spin and cis::Options, from BLAS, the spherical
# bridge and nonsymmetric_jk, so no device library. Its interface takes
# DeviceResources, a BasisSet, J's and K's classes and a SphericalMatrixAO,
# holds MatrixAOPairs and Buffers, returns Status and carries DavidsonOptions,
# hence everything named there PUBLIC. The implementation unit alone calls the
# gemms and nonsymmetric_jk, copies the scalars and checks the LinearOperator
# concept, so those and the FBF_TRY header are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.cis.sigma
  PRIMARY_INTERFACE
  sigma.cppm
  IMPLEMENTATION
  sigma.cpp
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
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::linear_operator
  free_breakfast::integrals::eri::integral_direct::nonsymmetric_jk
)
