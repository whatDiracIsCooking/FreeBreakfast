# The Hessian step. geometry, pipeline::gradient and pipeline::casscf_gradient
# are PUBLIC and re-exported: the product holds geometry::Vibrations, and the
# two gradient steps' modules re-export every step the two chains rerun, each
# a member, and MoleculeOutput, which run reads. core::resources and
# pipeline::step are in run's signature.
# run is defined in the implementation unit, so the analytic Hessian
# and the dipole derivative (hessian), the dipole integrals' limit
# (one_electron::dipole) and the basis's shells it is checked against (basis),
# the copy of C it takes (core::wwr), the reference orbitals'
# orthonormalization (recipes::blas, recipes::orthogonalize, core::constants),
# the planner it nests
# (pipeline::pipeline), the file it writes (io::json), core::log and
# diagnostics_macros (FBF_TRY, FBF_LOG_INFO) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  IMPLEMENTATION
  hessian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::geometry
  free_breakfast::pipeline::step
  free_breakfast::pipeline::gradient
  free_breakfast::pipeline::casscf_gradient
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::basis
  free_breakfast::hessian
  free_breakfast::integrals::one_electron::dipole
  free_breakfast::io::json
  free_breakfast::pipeline::pipeline
  free_breakfast::recipes::blas
  free_breakfast::recipes::orthogonalize
)
