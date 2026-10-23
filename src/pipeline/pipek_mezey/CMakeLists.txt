# The PipekMezey step. localization::pm, core::wwr and pipeline::scf are
# PUBLIC and re-exported: the product holds Buffers, the options are
# PipekMezeyOptions, and run's View names the Scf and the products it
# re-exports. core::resources, basis and pipeline::step are in run's
# signature, and run's inline body calls pipek_mezey_localize from the
# importer's TU. diagnostics_macros is only FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.pipek_mezey
  PRIMARY_INTERFACE
  pipek_mezey.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::basis
  free_breakfast::localization::pm
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
