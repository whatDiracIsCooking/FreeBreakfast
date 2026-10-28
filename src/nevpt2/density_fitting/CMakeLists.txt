# The slab permute. Device code in its own library, so its language and -x hip
# never reach the module compiles. The bridge header it shares with the module
# names int64_t, wwrStream_t and wwrError_t, hence types_headers and
# wwr::device PUBLIC; block_params.h (kFourWarps) and primitives.h
# (math::idivup) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.nevpt2.density_fitting.device
  SOURCES
  nevpt2_permute.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: nevpt2's MO blocks, h2e and slabs. DeviceResources, Buffers and
# Result are in its signatures, so core::resources, core::wwr and
# core::error_handling are PUBLIC. b_mo, the gemm's constants, the FBF_TRY
# header and the device library are the implementation unit's alone.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.density_fitting
  PRIMARY_INTERFACE
  density_fitting.cppm
  IMPLEMENTATION
  density_fitting.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::nevpt2::density_fitting::device
)
