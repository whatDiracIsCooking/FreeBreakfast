# The phase_scale kernel. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; parallel_for is its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.expm.device
  SOURCES
  expm.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# The module. DeviceResources and wwr's usual_fp / ComplexToRealType /
# RealToComplexType are in its signatures, so core::resources and core::wwr
# are PUBLIC. kImaginaryUnit and kOne / kZero are the implementation unit's
# alone, so core::constants is PRIVATE, as is the device library: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.expm
  PRIMARY_INTERFACE
  expm.cppm
  IMPLEMENTATION
  expm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::expm::device
)
