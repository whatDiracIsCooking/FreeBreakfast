# The S launcher. Device code in its own library, so its language and -x hip
# never reach the module compiles. Its sources are the launcher and the
# generated kernels (kernels/), one per class, each compiled on its own: the
# launcher includes only kernels/overlap.h and launches through its table.
# Under CUDA that wants relocatable device code, this library's default, so it
# must never pass NO_DEVICE_LINKING (nvcc warns #20280 without it). The bridge
# header it shares with the module names int32_t, kMaxAngularMomentum,
# PrimitivePairClassView, wwrStream_t and wwrError_t, hence types_headers,
# basis::headers, mmd::headers and wwr::device PUBLIC. The kernels include
# FelizJueves's felizjueves/runtime.h, which nothing outside this library sees,
# hence felizjueves::headers PRIVATE; so is the launcher's block sizing
# (block_params.h, primitives.h), hence core::kernel_headers.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.overlap.device
  SOURCES
  overlap_launch.cu
  kernels/s_00.cu
  kernels/s_01.cu
  kernels/s_02.cu
  kernels/s_11.cu
  kernels/s_12.cu
  kernels/s_22.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
)

# The module. DeviceResources, PrimitivePairClass and CartesianMatrixAO
# are in its signature, so core::resources, one_electron::upload and
# linalg::matrix_ao are PUBLIC. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.overlap
  PRIMARY_INTERFACE
  overlap.cppm
  IMPLEMENTATION
  overlap.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::one_electron::upload
  free_breakfast::integrals::mmd
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::overlap::device
)
