# The Guess step. basis (re-exported, and re-exporting molecule), taskflow
# (re-exported, for tf::Subflow) and pipeline::step are PUBLIC, since their
# types are in run's signature. sad is PUBLIC too: run's inline body calls
# sad_density from the importer's TU. diagnostics_macros is only FBF_TRY,
# expanded here. No GPU link: the step is host work.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.guess
  PRIMARY_INTERFACE
  guess.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  free_breakfast::third_party::taskflow
  free_breakfast::sad
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
