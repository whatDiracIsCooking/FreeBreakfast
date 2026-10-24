# The mixed S launcher. Device code in its own library, so its language and
# -x hip never reach the module compiles. Its sources are the launcher and the
# generated kernels (kernels/), one per ORDERED class, each compiled on its own:
# the launcher includes only kernels/overlap_mixed.h and launches through its
# table. Under CUDA that wants relocatable device code, this library's default,
# so it must never pass NO_DEVICE_LINKING. The bridge header it shares with the
# module names int32_t, kMaxAngularMomentum, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, basis::headers,
# mmd::headers and wwr::device PUBLIC. The kernels include FelizJueves's
# felizjueves/runtime.h, hence felizjueves::headers PRIVATE; so is the
# launcher's block sizing, hence core::kernel_headers.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.overlap_mixed.device
  SOURCES
  overlap_mixed_launch.cu
  kernels/sx_00.cu
  kernels/sx_01.cu
  kernels/sx_02.cu
  kernels/sx_10.cu
  kernels/sx_11.cu
  kernels/sx_12.cu
  kernels/sx_20.cu
  kernels/sx_21.cu
  kernels/sx_22.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
)

# The module. DeviceResources, BasisSet and the device buffer are in its
# signature, so core::resources, basis and core::wwr are PUBLIC. It builds the
# records (mmd), launches through launch_each (one_electron::launch) and
# scales the spherical form's gemms (core::constants) in its
# implementation only, so those are PRIVATE, as is the device library: a static
# library's private link dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.overlap_mixed
  PRIMARY_INTERFACE
  overlap_mixed.cppm
  IMPLEMENTATION
  overlap_mixed.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::integrals::mmd
  free_breakfast::integrals::one_electron::launch
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::overlap_mixed::device
)
