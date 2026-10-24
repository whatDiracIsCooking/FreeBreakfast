# The host reference sigma: written for clarity, it is the oracle the GPU sigma
# is tested against. ci::strings is PUBLIC because the interface takes
# StringSets, core::error_handling because it returns a Status.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.sigma
  PRIMARY_INTERFACE
  sigma.cppm
  IMPLEMENTATION
  sigma.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::ci::strings
)
