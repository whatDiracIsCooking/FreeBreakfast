# Host-side over the strings' lists. ci::strings is PUBLIC because the
# interface takes StringSets, core::error_handling because it returns a Status.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.rdm
  PRIMARY_INTERFACE
  rdm.cppm
  IMPLEMENTATION
  rdm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::ci::strings
)
