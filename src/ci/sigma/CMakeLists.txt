# The GPU sigma's two gathers and the diagonal. Device code in its own library,
# so its language and -x hip never reach the module compiles. The bridge header
# it shares with the module names int32_t, uint64_t, wwrStream_t and
# wwrError_t, hence types_headers and wwr::device PUBLIC; block_params.h
# (kFourWarps) and primitives.h (math::idivup), both from
# core::kernel_headers, are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.ci.sigma.device
  SOURCES
  ci_sigma.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: ci::Sigma, ci::unpack and ci::pack. ci::strings is PUBLIC because
# the interface takes ci::StringSets, core::error_handling because it returns a
# Status, core::wwr and core::resources because ci::Sigma holds Buffers and
# takes DeviceResources. The implementation unit alone GMF-includes the bridge,
# calls the gemm and checks the LinearOperator concept, so the device library,
# constants, primitives, the FBF_TRY header and recipes::linear_operator are
# PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.sigma
  PRIMARY_INTERFACE
  sigma.cppm
  IMPLEMENTATION
  sigma.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::ci::strings
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::linear_operator
  free_breakfast::ci::sigma::device
)
