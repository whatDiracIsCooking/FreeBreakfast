# The module. Host code over wwr's BLAS, with no kernels of its own: A^T S C
# is two gemms and the per-atom sums are host work over m x n_orb. atom and
# basis are in the signatures, so PUBLIC; the IAOs are an argument, so
# localization::ibo is no dependency at all.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.analysis.population
  PRIMARY_INTERFACE
  population.cppm
  IMPLEMENTATION
  population.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::atom
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
