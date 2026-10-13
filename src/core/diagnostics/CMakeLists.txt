# diagnostics.h is a PLAIN header, not a module -- macros (FBF_TRY,
# FBF_GPU_MUST, FBF_LOG_*) cannot cross a module boundary. A consumer includes
# it after importing the modules the macros expand against, rather than
# importing it. LINK_PUBLIC is not for the macros themselves -- the header
# expects those names in scope from the includer's own imports -- it just makes
# linking this one target enough to be able to write those imports:
# error_handling (check_or_abort, gpu_status) and, through it, log
# (log::should_log).
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.diagnostics_macros
  LINK_PUBLIC
  free_breakfast::core::error_handling
)
