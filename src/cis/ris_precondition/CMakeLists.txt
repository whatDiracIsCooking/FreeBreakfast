# The module: cis::RisPreconditioner, from BLAS, the density-fitting chain,
# lu_solve and cis.precondition's diagonal-correction launcher, so no device
# library of its own. Its interface takes DeviceResources, a BasisSet, the
# pair classes, the atoms, a SphericalMatrixAO and RisOptions, holds Buffers
# and returns Status, hence everything named there PUBLIC. The implementation
# unit alone GMF-includes the precondition bridge, fits B, builds L (and the
# exchange's T_oo and T_vv), runs the gemms, lu_solve and the GMRES, aligns
# the workspace's regions, copies the scalars and checks the
# DavidsonPreconditioner concept, so those and the FBF_TRY header are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.cis.ris_precondition
  PRIMARY_INTERFACE
  ris_precondition.cppm
  IMPLEMENTATION
  ris_precondition.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::atom
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::recipes::matrix_ao
  free_breakfast::cis::ris_basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::davidson
  free_breakfast::recipes::lu
  free_breakfast::integrals::eri::density_fitting
  free_breakfast::cis::precondition::device
)
