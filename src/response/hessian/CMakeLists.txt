# The module: response::Hessian, declared in the interface and defined in the
# implementation unit. BasisSet, DeviceResources, Buffers, the two kinds of
# pair class, the AO matrices and response::Fock are in its signature or
# members, so basis, core::wwr, core::resources, coulomb, exchange,
# recipes::matrix_ao and response::fock are PUBLIC; the density-fitted J and K
# are only called, so jk is PRIVATE. It launches nothing of its own: J and K
# are coulomb's and exchange's or df_coulomb's and df_exchange's, the rest BLAS
# and copies, so there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.response.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  IMPLEMENTATION
  hessian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::response::fock
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::linear_operator
  free_breakfast::recipes::norm
  free_breakfast::integrals::eri::density_fitting::jk
)
