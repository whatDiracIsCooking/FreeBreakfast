add_subdirectory(diis)
add_subdirectory(orthogonalize)

# The aggregate: one import for the dense linear-algebra layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::linalg::diis
  free_breakfast::linalg::orthogonalize
)
