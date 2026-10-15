# One class composing core::wwr's aliases, defined inline, so every name it
# hands out is instantiated in the consumer's TU: every link is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.device_resources
  PRIMARY_INTERFACE
  device_resources.cppm
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::nontransferrable
  free_breakfast::core::wwr
  IMPORT_STD
)
