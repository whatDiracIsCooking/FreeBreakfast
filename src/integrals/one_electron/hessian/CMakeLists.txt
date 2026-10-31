# The Hessian launchers. Device code in its own library, so its language and -x
# hip never reach the module compiles. Its sources are the launcher and the
# generated kernels (kernels/), three families of one unit per class, each
# compiled on its own: the launcher includes only the families' headers and
# launches through their tables. Under CUDA that wants relocatable device code,
# this library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns
# #20280 without it). The bridge header it shares with the module names int32_t,
# PrimitivePairClassView, NucleiView, wwrStream_t and wwrError_t, hence
# types_headers, mmd::headers, molecule::headers and wwr::device PUBLIC. The
# kernels include FelizJueves's felizjueves/runtime.h and (d2v) the shared
# eval_boys.h, which nothing outside this library sees, hence
# felizjueves::headers and shared::headers PRIVATE; so is the launcher's block
# sizing (block_params.h, primitives.h), hence core::kernel_headers.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.hessian.device
  SOURCES
  hessian_launch.cu
  kernels/d2s_00.cu
  kernels/d2s_01.cu
  kernels/d2s_02.cu
  kernels/d2s_03.cu
  kernels/d2s_11.cu
  kernels/d2s_12.cu
  kernels/d2s_13.cu
  kernels/d2s_22.cu
  kernels/d2s_23.cu
  kernels/d2s_33.cu
  kernels/d2t_00.cu
  kernels/d2t_01.cu
  kernels/d2t_02.cu
  kernels/d2t_03.cu
  kernels/d2t_11.cu
  kernels/d2t_12.cu
  kernels/d2t_13.cu
  kernels/d2t_22.cu
  kernels/d2t_23.cu
  kernels/d2t_33.cu
  kernels/d2v_00.cu
  kernels/d2v_01.cu
  kernels/d2v_02.cu
  kernels/d2v_03.cu
  kernels/d2v_11.cu
  kernels/d2v_12.cu
  kernels/d2v_13.cu
  kernels/d2v_22.cu
  kernels/d2v_23.cu
  kernels/d2v_33.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  free_breakfast::molecule::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)

# The module. DeviceResources, BasisSet, PrimitivePairClass, Molecule and
# SphericalMatrixAO are in its signatures, so core::resources, basis, mmd,
# molecule and recipes::matrix_ao are PUBLIC. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launchers, and a static library's private link dependencies still reach the
# final link. one_electron::launch is PRIVATE too: only the implementation
# runs its checks and launch_each.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  IMPLEMENTATION
  hessian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::one_electron::hessian::device
)
