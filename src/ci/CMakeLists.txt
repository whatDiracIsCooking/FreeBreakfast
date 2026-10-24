# Determinant CI over plain arrays: CASCI, CASSCF and FCI all call it, so it
# never sees a basis, molecule or SCF. Sigma is over the strings' lists.
add_subdirectory(strings)
add_subdirectory(sigma)

# The aggregate: one import for the CI layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci
  PRIMARY_INTERFACE
  ci.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
)
