# The Localization step. localization::ibo (with the pipek_mezey it
# re-exports), analysis::population, core::wwr and pipeline::scf are PUBLIC
# and re-exported: the
# product holds Buffers, the options are PipekMezeyOptions, and run's View
# names the Scf and the products it re-exports. core::resources, basis and
# pipeline::step are in run's signature. run is defined in the
# implementation unit, so io::basis (MINAO), the Molden writer and
# diagnostics_macros (FBF_TRY) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.localization
  PRIMARY_INTERFACE
  localization.cppm
  IMPLEMENTATION
  localization.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::basis
  free_breakfast::localization::pipek_mezey
  free_breakfast::localization::ibo
  free_breakfast::analysis::population
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::atom
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::basis
  free_breakfast::io::molden
)
