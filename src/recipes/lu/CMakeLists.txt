# The module. DeviceResources and MemoryFootprint are in its signatures, so
# core::resources and core::footprint are PUBLIC. getrf and getrs are wwr's,
# called only from the implementation unit, so core::wwr is PRIVATE, as is
# core::primitives (the footprint's align_up).
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.lu
  PRIMARY_INTERFACE
  lu.cppm
  IMPLEMENTATION
  lu.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::footprint
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
