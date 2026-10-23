# The TOML front end and the step registry. The pipeline and every step module
# are PUBLIC: Calculation, what parse_config returns, names them all. This is
# the one library that imports every step; only it and main pay that build
# fan-in. io::toml is PRIVATE -- only the implementation unit walks the
# document, so no importer compiles toml++.
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
  free_breakfast::pipeline::pipeline
  free_breakfast::pipeline::molecule
  free_breakfast::pipeline::basis_set
  free_breakfast::pipeline::spherical_transform
  free_breakfast::pipeline::primitive_pair_classes
  free_breakfast::pipeline::overlap
  free_breakfast::pipeline::orthogonalizer
  free_breakfast::pipeline::core_hamiltonian
  free_breakfast::pipeline::guess
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::io::base
  free_breakfast::io::basis
  free_breakfast::io::toml
  free_breakfast::core::diagnostics_macros
)
