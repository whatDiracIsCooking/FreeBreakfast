# The diis_solve kernel. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; block_params.h (kWarpSize) and primitives.h (math::fabs)
# are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.diis.device
  SOURCES
  diis_solve.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources is in its signatures and Buffers in its class
# layout, so core::resources and core::wwr are PUBLIC. The device library is
# PRIVATE: the implementation unit includes the bridge header in its GMF and
# calls the launcher, and a static library's private link dependencies still
# reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.diis
  PRIMARY_INTERFACE
  diis.cppm
  IMPLEMENTATION
  diis.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::diis::device
)
