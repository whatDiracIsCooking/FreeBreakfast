# The diagonal slices and the Sijrs reduction. Device code in its own library,
# so its language and -x hip never reach the module compiles. The bridge
# header it shares with the module names int32_t, wwrStream_t and wwrError_t,
# hence types_headers and wwr::device PUBLIC; block_reduce.cuh (with
# kFourWarps and math::idivup) is its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.nevpt2.energy.device
  SOURCES
  nevpt2_energy.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources, MoBlocks and Result are in its signatures, so
# core::resources, nevpt2::density_fitting and core::error_handling are PUBLIC.
# The einsum, the solver, BLAS, the FBF_TRY header and the device library are
# the implementation units' alone.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.energy
  PRIMARY_INTERFACE
  energy.cppm
  PARTITIONS
  shared.cppm
  IMPLEMENTATION
  energy.cpp
  pc.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::nevpt2::density_fitting
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::einsum
  free_breakfast::nevpt2::energy::device
)
