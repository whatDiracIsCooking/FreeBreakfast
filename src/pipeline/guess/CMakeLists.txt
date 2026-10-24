# The Guess step. The basis_set step (re-exported, and re-exporting the
# molecule step), taskflow (re-exported, for tf::Subflow) and pipeline::step
# are PUBLIC, since their types are in run's signature. sad and io::sad are
# PUBLIC too: run's inline body calls sad_density, passing
# builtin_sad_density, from the importer's TU,
# and io::npy for the same reason, load's inline body calling read_npy.
# diagnostics_macros is only FBF_TRY, expanded here. No GPU link: the step is
# host work.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.guess
  PRIMARY_INTERFACE
  guess.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::pipeline::basis_set
  free_breakfast::third_party::taskflow
  free_breakfast::sad
  free_breakfast::io::sad
  free_breakfast::io::npy
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
