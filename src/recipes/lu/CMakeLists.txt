# The row-interchange kernel. Device code in its own library, so its language
# and -x hip never reach the module compiles. The bridge header it shares with
# the module names int32_t, int64_t, wwrStream_t and wwrError_t, hence
# types_headers and wwr::device PUBLIC; block_params.h (kFourWarps) and
# primitives.h (idivup) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.lu.device
  SOURCES
  lu.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources and MemoryFootprint are in its signatures, so
# core::resources and core::footprint are PUBLIC. getrf is wwr's and the
# triangular solves the trsm recipe's, called only from the implementation
# unit, so core::wwr and recipes::trsm are PRIVATE, as are core::primitives
# (the footprint's align_up) and the device library, reached through the
# bridge header in the implementation unit's GMF.
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
  free_breakfast::recipes::lu::device
  free_breakfast::recipes::trsm
)
