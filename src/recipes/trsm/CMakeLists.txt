# The diagonal-block kernel. Device code in its own library, so its language
# and -x hip never reach the module compiles. The bridge header it shares with
# the module names int32_t, int64_t, wwrStream_t and wwrError_t, hence
# types_headers and wwr::device PUBLIC; block_params.h (kFourWarps) and
# primitives.h (idivup) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.trsm.device
  SOURCES
  trsm.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources and wwr's side, fill, op and diag enums are in
# its signature, so core::resources and core::wwr are PUBLIC. The device
# library is reached through the bridge header in the implementation unit's
# GMF, so it is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.trsm
  PRIMARY_INTERFACE
  trsm.cppm
  IMPLEMENTATION
  trsm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::trsm::device
)
