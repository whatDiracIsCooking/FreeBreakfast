# The launch staging the DF gradient and the DF Hessian share: their kernels
# read one parameter block. Internal to density fitting: the aggregate does
# not re-export it, and gradient and hessian link it PRIVATE. Its own module,
# not a partition, because a partition belongs to one module and these are
# two. DeviceResources, BasisSet, PrimitivePairClass and Result are in its
# signature, so core::resources, core::error_handling, basis and mmd are
# PUBLIC. The implementation unit includes both modules' bridge headers in its
# GMF for the record layouts it writes (and checks they agree), hence their
# device libraries PRIVATE; it launches nothing of its own, only BLAS and
# copies.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.staging
  PRIMARY_INTERFACE
  staging.cppm
  IMPLEMENTATION
  staging.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::eri::density_fitting::gradient::device
  free_breakfast::integrals::eri::density_fitting::hessian::device
)
