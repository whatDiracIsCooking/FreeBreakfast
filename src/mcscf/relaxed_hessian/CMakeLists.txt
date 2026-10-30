# The module: mcscf::RelaxedHessian and the orbital-CI coupling, declared in
# the interface and defined in the implementation unit. Internal to the MCSCF
# layer: free_breakfast.mcscf does not re-export it, and solve links it
# PRIVATE. DeviceResources, Buffers, the strings, the sigma, the Rdm, ci::Roots
# and response::Hessian are in its signature or members, so core::wwr,
# core::resources, ci::strings, ci::sigma, ci::rdm, ci::solver and
# response::hessian are PUBLIC; the packed order's copies and FBF_TRY are only
# used. It launches nothing of its own: sigmas, RDMs, H's apply, BLAS and
# copies, so there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.relaxed_hessian
  PRIMARY_INTERFACE
  relaxed_hessian.cppm
  IMPLEMENTATION
  relaxed_hessian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::ci::solver
  free_breakfast::response::hessian
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::response::orbital_pairs
)
