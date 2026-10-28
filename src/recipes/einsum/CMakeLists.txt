# The contraction kernel. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; block_params.h (kFourWarps) and primitives.h (idivup) are
# its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.einsum.device
  SOURCES
  einsum.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. wwrStream_t and Status are in its signatures, so core::wwr and
# core::error_handling are PUBLIC. The device library is reached through the
# bridge header in the implementation unit's GMF, so it is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.einsum
  PRIMARY_INTERFACE
  einsum.cppm
  IMPLEMENTATION
  einsum.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::einsum::device
)
