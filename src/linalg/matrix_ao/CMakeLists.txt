# The module. DeviceResources is in its signatures, Buffers and NonCopyable in
# its class layouts and BasisSet in its constructors, so core::resources,
# core::wwr, core::nontransferrable and basis are PUBLIC. No device library:
# both products are gemmwich, from linalg::blas (PRIVATE: only the
# implementation unit calls it).
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg.matrix_ao
  PRIMARY_INTERFACE
  matrix_ao.cppm
  IMPLEMENTATION
  matrix_ao.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::nontransferrable
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::linalg::blas
)
