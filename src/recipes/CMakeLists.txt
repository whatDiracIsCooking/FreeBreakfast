add_subdirectory(blas)
add_subdirectory(complex_cast)
add_subdirectory(diagonalize)
add_subdirectory(diis)
add_subdirectory(matrix_ao)
add_subdirectory(orthogonalize)

# The aggregate: one import for the dense linear-algebra layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes
  PRIMARY_INTERFACE
  recipes.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::recipes::blas
  free_breakfast::recipes::complex_cast
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::diis
  free_breakfast::recipes::matrix_ao
  free_breakfast::recipes::orthogonalize
)
