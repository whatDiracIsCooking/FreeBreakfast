# The TOML front end. The step modules are PUBLIC: their types are what
# parse_config returns. io::toml is PRIVATE -- only the implementation unit
# walks the document, so no importer compiles toml++.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.config
  PRIMARY_INTERFACE
  config.cppm
  IMPLEMENTATION
  config.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::pipeline::molecule
  free_breakfast::pipeline::basis_set
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::io::base
  free_breakfast::io::toml
  free_breakfast::core::diagnostics_macros
)
