# The split_planes / merge_planes kernels. Device code in its own library, so
# its language and -x hip never reach the module compiles. The bridge header it
# shares with the module names int32_t, wwrStream_t and wwrError_t, hence
# types_headers and wwr::device PUBLIC; parallel_for is its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.complex_cast.device
  SOURCES
  complex_cast.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# The module. DeviceResources and wwr's complex_fp / ComplexToRealType are in
# its signatures, so core::resources and core::wwr are PUBLIC. The device
# library is PRIVATE: the implementation unit includes the bridge header in its
# GMF and calls the launchers, and a static library's private link dependencies
# still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.complex_cast
  PRIMARY_INTERFACE
  complex_cast.cppm
  IMPLEMENTATION
  complex_cast.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::recipes::complex_cast::device
)
