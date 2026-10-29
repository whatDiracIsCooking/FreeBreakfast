# The orbital-energy diagonal and the preconditioner's correction. Device code
# in its own library, so its language and -x hip never reach the module
# compiles. The bridge header it shares with the module names int32_t,
# wwrStream_t and wwrError_t, hence types_headers and wwr::device PUBLIC;
# block_params.h (kFourWarps) and primitives.h (math::idivup), both from
# core::kernel_headers, are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.cis.precondition.device
  SOURCES
  cis_precondition.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: cis::diagonal, cis::guess and cis::Preconditioner. Its interface
# takes DeviceResources and returns Status, and Preconditioner holds a
# Buffers::device, so core::resources, core::error_handling and core::wwr are
# PUBLIC. The implementation unit alone GMF-includes the bridge, copies kOne
# and checks the DavidsonPreconditioner concept, so the device library,
# core::constants, recipes::davidson and the FBF_TRY header are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.cis.precondition
  PRIMARY_INTERFACE
  precondition.cppm
  IMPLEMENTATION
  precondition.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::davidson
  free_breakfast::cis::precondition::device
)
