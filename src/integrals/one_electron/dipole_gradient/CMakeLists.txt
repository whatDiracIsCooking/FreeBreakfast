# The dipole-gradient launcher. Device code in its own library, so its language
# and -x hip never reach the module compiles. Its sources are the launcher and
# the generated kernels (kernels/), one unit per class through (ff), each
# compiled on its own: the launcher includes only kernels/dipole_gradient.h and
# launches through its table. Under CUDA that wants relocatable device code,
# this library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns
# #20280 without it). The bridge header it shares with the module names
# int32_t, PrimitivePairClassView, wwrStream_t and wwrError_t, hence
# types_headers, mmd::headers and wwr::device PUBLIC. The kernels include
# FelizJueves's felizjueves/runtime.h, which nothing outside this library sees,
# hence felizjueves::headers PRIVATE; so are the launcher's block sizing
# (block_params.h, primitives.h) and its origin and charge (constants.h), hence
# core::kernel_headers and core::constants_header.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.dipole_gradient.device
  SOURCES
  dipole_gradient_launch.cu
  kernels/gmu_00.cu
  kernels/gmu_01.cu
  kernels/gmu_02.cu
  kernels/gmu_03.cu
  kernels/gmu_11.cu
  kernels/gmu_12.cu
  kernels/gmu_13.cu
  kernels/gmu_22.cu
  kernels/gmu_23.cu
  kernels/gmu_33.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
  felizjueves::headers
)

# The module. DeviceResources, BasisSet, PrimitivePairClass, SphericalMatrixAO
# and Result are in its signature, so core::resources, basis, mmd,
# recipes::matrix_ao and core::error_handling are PUBLIC. The device library is
# PRIVATE: the implementation unit includes the bridge header in its GMF and
# calls the launcher, and a static library's private link dependencies still
# reach the final link. one_electron::launch is PRIVATE too: only the
# implementation runs its checks and launch_each.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.dipole_gradient
  PRIMARY_INTERFACE
  dipole_gradient.cppm
  IMPLEMENTATION
  dipole_gradient.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::one_electron::dipole_gradient::device
)
