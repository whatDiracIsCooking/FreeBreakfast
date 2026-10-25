# Everything PUBLIC: davidson_solve is a template over the caller's operator,
# instantiated in the consumer's TU, so what it names -- wwr's BLAS and
# ScopedPointerMode (core::wwr), kOne / kNegOne (core::constants), the log
# sink, diagonalize, orthonormalize -- must reach it. The FBF_TRY /
# FBF_LOG_* macros expand in this interface unit, so their header is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.davidson
  PRIMARY_INTERFACE
  davidson.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::constants
  free_breakfast::core::error_handling
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::linear_operator
  free_breakfast::recipes::orthogonalize
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
