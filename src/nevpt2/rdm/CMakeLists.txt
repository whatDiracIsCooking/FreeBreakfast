# nevpt2::rdms' device half: the eri's conversion, R and L2 per tile, fdm2 and
# the wedge. Device code in its own library, so its language and -x hip never
# reach the module compiles. The bridge header it shares with the module names
# int32_t, Replacement, wwrStream_t and wwrError_t, hence types_headers,
# ci::strings::headers and wwr::device PUBLIC; block_params.h (kFourWarps) and
# primitives.h (math::idivup), from core::kernel_headers, are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.nevpt2.rdm.device
  SOURCES
  nevpt2_rdm.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::ci::strings::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: nevpt2::rdms. ci::strings is PUBLIC because the interface takes
# DeviceStringSets, core::error_handling because it returns a Result, and
# core::resources because it takes DeviceResources. The implementation unit
# alone GMF-includes the bridge and calls the gemms, so the device library,
# core::wwr, constants, primitives and the FBF_TRY header are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.rdm
  PRIMARY_INTERFACE
  rdm.cppm
  IMPLEMENTATION
  rdm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::ci::strings
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::nevpt2::rdm::device
)
