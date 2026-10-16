# DeviceResources is defined inline and composes core::wwr's aliases, so every
# name it hands out is instantiated in the consumer's TU: those links are
# PUBLIC. sys_env's interface names only std and free_breakfast types; what its
# implementation alone reaches (log, the FBF_* macros) is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.resources
  PRIMARY_INTERFACE
  resources.cppm
  PARTITIONS
  device_resources.cppm
  sys_env.cppm
  IMPLEMENTATION
  sys_env.cpp
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::nontransferrable
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  IMPORT_STD
)
