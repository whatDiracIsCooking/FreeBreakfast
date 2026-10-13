# log is PUBLIC because check_or_abort is a template: its body (log::critical)
# is instantiated in the consumer's TU, so the consumer must be able to reach
# that name too.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.error_handling
  PRIMARY_INTERFACE
  error_handling.cppm
  PARTITIONS
  error.cppm
  abort.cppm
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::log
  IMPORT_STD
)
