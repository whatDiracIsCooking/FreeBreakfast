# The module. Host arithmetic over orbital weights, no GPU: the weights come
# from analysis::population's orbital_weights, but as a plain span, so that
# is no dependency. basis is in the signature, so PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.analysis.bond
  PRIMARY_INTERFACE
  bond.cppm
  IMPLEMENTATION
  bond.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
)
