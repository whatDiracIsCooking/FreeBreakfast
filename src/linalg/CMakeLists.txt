add_subdirectory(blas)
add_subdirectory(diagonalize)
add_subdirectory(diis)
add_subdirectory(matrix_ao)
add_subdirectory(orthogonalize)

# The aggregate: one import for the dense linear-algebra layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::linalg::blas
  free_breakfast::linalg::diagonalize
  free_breakfast::linalg::diis
  free_breakfast::linalg::matrix_ao
  free_breakfast::linalg::orthogonalize
)
