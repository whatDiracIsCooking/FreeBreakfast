# CiRdm's gather of D and Γ's assembly. Device code in its own library, so
# its language and -x hip never reach the module compiles. The bridge header it
# shares with the module names int32_t, wwrStream_t and wwrError_t, hence
# types_headers and wwr::device PUBLIC; block_params.h (kFourWarps) and
# primitives.h (math::idivup), both from core::kernel_headers, are its own
# business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.ci.rdm.device
  SOURCES
  ci_rdm.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: rdm1, host_rdm12 and CiRdm. ci::strings is PUBLIC because the
# interface takes StringSets, core::error_handling because it returns a
# Status, core::resources because CiRdm takes DeviceResources, and core::wwr
# because CiRdm holds Buffers. The implementation unit alone GMF-includes the
# bridge, so the device library, constants, primitives and the FBF_TRY header
# are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.rdm
  PRIMARY_INTERFACE
  rdm.cppm
  IMPLEMENTATION
  rdm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::ci::strings
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::ci::rdm::device
)
