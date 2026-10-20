# DeviceResources and NodeResources' accessors are defined inline and compose
# core::wwr's aliases and taskflow's Executor, so every name they hand out is
# instantiated in the consumer's TU: those links are PUBLIC. What only the
# implementation units reach (log, the FBF_* macros) is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.resources
  PRIMARY_INTERFACE
  resources.cppm
  PARTITIONS
  device_resources.cppm
  node_resources.cppm
  sys_env.cppm
  IMPLEMENTATION
  node_resources.cpp
  sys_env.cpp
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::nontransferrable
  free_breakfast::core::primitives
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::third_party::taskflow
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  IMPORT_STD
)
