# Host-only and over plain integers: no device code, no DeviceResources. The
# GPU copies of these tables come with the GPU sigma, in its own target.
# core::error_handling is PUBLIC because create returns a Result.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.strings
  PRIMARY_INTERFACE
  strings.cppm
  IMPLEMENTATION
  strings.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
)
