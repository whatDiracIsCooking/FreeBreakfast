# The diagonal-block kernel. Device code in its own library, so its language
# and -x hip never reach the module compiles. The bridge header it shares with
# the module names int32_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; block_params.h (kFourWarps) is its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.cholesky.device
  SOURCES
  cholesky.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources is in its signature, so core::resources is
# PUBLIC. trsm and syrk are wwr's, and the device library is reached through
# the bridge header in the implementation unit's GMF, so both are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.cholesky
  PRIMARY_INTERFACE
  cholesky.cppm
  IMPLEMENTATION
  cholesky.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::cholesky::device
)
