# The kernels, their launchers and the arithmetic they run. Device code in its
# own library, so its language and -x hip never reach the module compiles. The
# bridge header names int32_t, wwrStream_t and wwrError_t, hence types_headers
# and wwr::device PUBLIC; block_reduce, the launch shapes (kernel_headers) and
# math:: (primitives_header) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.localization.pipek_mezey.device
  SOURCES
  pipek_mezey_kernels.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::primitives_header
)

# The module. pipek_mezey_localize is declared in the interface and defined in
# the implementation unit, which alone GMF-includes the bridge. So the device
# library and what only the driver uses are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.localization.pipek_mezey
  PRIMARY_INTERFACE
  pipek_mezey.cppm
  IMPLEMENTATION
  pipek_mezey.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::localization::pipek_mezey::device
)
