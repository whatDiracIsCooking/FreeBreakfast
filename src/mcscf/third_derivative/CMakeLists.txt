# The module: mcscf::ThirdDerivative, declared in the interface and defined in
# the implementation unit. DeviceResources, Buffers, the AO matrices, the
# strings, the sigma and the Rdm are in its signature or members, so
# core::wwr, core::resources, recipes::matrix_ao, ci::strings, ci::sigma and
# ci::rdm are PUBLIC; the packed order's copies, the gemms and FBF_TRY are only
# used. It launches nothing of its own: sigmas, RDMs, BLAS and copies, so
# there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.third_derivative
  PRIMARY_INTERFACE
  third_derivative.cppm
  IMPLEMENTATION
  third_derivative.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::matrix_ao
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::response::orbital_pairs
  free_breakfast::recipes::blas
)
