# The module: mcscf::zvector, declared in the interface and defined in the
# implementation unit. Internal to the MCSCF layer: free_breakfast.mcscf does
# not re-export it, and a consumer (the SA-CASSCF gradient) links it PRIVATE.
# DeviceResources, Buffers, ci::Roots, response::Hessian and RelaxedHessian
# are in its signature or result, so core::wwr, core::resources, ci::solver,
# response::hessian and mcscf::relaxed_hessian are PUBLIC; the BLAS scalars,
# the projections' gemms and FBF_TRY are only used. It launches nothing of its
# own: H~'s products, BLAS and copies, so there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.zvector
  PRIMARY_INTERFACE
  zvector.cppm
  IMPLEMENTATION
  zvector.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::ci::solver
  free_breakfast::response::hessian
  free_breakfast::mcscf::relaxed_hessian
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
)
