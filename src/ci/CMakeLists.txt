# Determinant CI over plain arrays: CASCI, CASSCF and FCI all call it, so it
# never sees a basis, molecule or SCF. One member so far, the strings.
add_subdirectory(strings)

# The aggregate: one import for the CI layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci
  PRIMARY_INTERFACE
  ci.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::ci::strings
)
