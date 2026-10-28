# The two tile kernels. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t, int64_t, wwrStream_t and wwrError_t, hence
# types_headers and wwr::device PUBLIC; block_params.h and primitives.h (from
# core::kernel_headers) and constants.h (kHalf, from core::constants_header)
# are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.symmetrize.device
  SOURCES
  symmetrize.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
)

# The module. DeviceResources and wwr's wwrblasFillMode_t are in its
# signatures, so core::resources and core::wwr are PUBLIC. The device library
# is reached through the bridge header in the implementation unit's GMF, so it
# is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.symmetrize
  PRIMARY_INTERFACE
  symmetrize.cppm
  IMPLEMENTATION
  symmetrize.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::recipes::symmetrize::device
)
