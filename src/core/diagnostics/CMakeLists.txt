# wwr::extension::common is PUBLIC because gpu_status is a template over
# wwr::extension::error_type: its body (success_code / error_name /
# error_string) is instantiated in the consumer's TU, so the consumer must be
# able to reach those names too.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.diagnostics
  PRIMARY_INTERFACE
  diagnostics.cppm
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::log
  wwr::extension::common
  IMPORT_STD
)

# diagnostics.h is a PLAIN header, not a module partition -- macros (FBF_TRY,
# FBF_GPU_MUST, FBF_LOG_*) cannot cross a module boundary. It is a separate
# target from the module above so a consumer includes it (after importing the
# modules the macros expand against) rather than importing it. LINK_PUBLIC on
# the module targets is not for the macros themselves -- the header expects
# those names in scope from the includer's own imports -- it just keeps a single
# "give me diagnostics" dependency working whether a caller reaches for the
# module API or the macro header.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.diagnostics_macros
  LINK_PUBLIC
  free_breakfast::core::diagnostics
)
