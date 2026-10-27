# The abs_at kernel. Device code in its own library, so its language and -x hip
# never reach the module compiles. The bridge header it shares with the module
# names int32_t, wwrStream_t and wwrError_t, hence types_headers and wwr::device
# PUBLIC; primitives.h (math::fabs) is its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.norm.device
  SOURCES
  abs_at.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::primitives_header
)

# The module. DeviceResources is in its signature, so core::resources is
# PUBLIC. iamax and ScopedPointerMode are wwr's, called only from the
# implementation unit, so core::wwr is PRIVATE, as is the device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.norm
  PRIMARY_INTERFACE
  norm.cppm
  IMPLEMENTATION
  norm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::norm::device
)
