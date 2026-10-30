# The module: host code over the one-electron and density-fitting derivative
# contractions and wwr's BLAS, with no kernels of its own. DeviceResources,
# Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO and Result are in
# its signatures, so core::resources, molecule, basis, mmd, recipes::matrix_ao
# and core::error_handling are PUBLIC. The derivative calls, b_mo, the gemms
# and the FBF_TRY header are the implementation units' own, so PRIVATE.
# rhf takes K's OrderedPairClass, so exchange::pairs is PUBLIC too; its
# coulomb_gradient and exchange_gradient are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  one_electron.cpp
  df_rhf.cpp
  direct_rhf.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::integrals::mmd
  free_breakfast::integrals::eri::integral_direct::exchange::pairs
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::one_electron::gradient
  free_breakfast::integrals::eri::density_fitting::gradient
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
)
