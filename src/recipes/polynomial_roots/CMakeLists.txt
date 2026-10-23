# The Aberth-Ehrlich kernel. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; kernel_headers (kWarpSize, warp_reduce, and through it
# primitives.h) and constants_header (kPi) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.polynomial_roots.device
  SOURCES
  aberth.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
)

# The module. DeviceResources and wwr's usual_fp / complex_fp /
# ComplexToRealType are in its signature, so core::resources and core::wwr are
# PUBLIC. The device library is PRIVATE: the implementation unit includes the
# bridge header in its GMF and calls the launcher, and a static library's
# private link dependencies still reach the final link. core::kernel is
# PRIVATE too: the implementation unit checks the degree against kWarpSize.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.polynomial_roots
  PRIMARY_INTERFACE
  polynomial_roots.cppm
  IMPLEMENTATION
  polynomial_roots.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::kernel
  free_breakfast::recipes::polynomial_roots::device
)
