# The interface names only std and free_breakfast types, so everything the
# implementation reaches -- the runtime API for device enumeration, log and the
# FBF_* macros -- is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.sys_env
  PRIMARY_INTERFACE
  sys_env.cppm
  IMPLEMENTATION
  sys_env.cpp
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  wwr::runtime_api
  IMPORT_STD
)
