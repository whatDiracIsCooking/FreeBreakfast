# ci::Rdm's gather of D and Γ's assembly. Device code in its own library, so
# its language and -x hip never reach the module compiles. The bridge header it
# shares with the module names int32_t, Replacement, wwrStream_t and
# wwrError_t, hence types_headers, ci::strings::headers and wwr::device PUBLIC;
# block_params.h (kFourWarps),
# primitives.h (math::idivup), both from core::kernel_headers, and the sigma's
# ci_symmetric.h, reached through ci::sigma::device's include directory, are
# its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.ci.rdm.device
  SOURCES
  ci_rdm.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::ci::strings::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::ci::sigma::device
)

# The module: ci::Rdm. ci::strings is PUBLIC because the interface takes
# DeviceStringSets, core::error_handling because it returns a Status,
# core::resources because ci::Rdm takes DeviceResources, and ci::sigma because
# rdm12 takes its ci::Layout. The implementation unit alone GMF-includes the
# bridge and calls the syrk and gemm, so the device library, core::wwr,
# constants and the FBF_TRY header are PRIVATE.
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
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::ci::rdm::device
)
