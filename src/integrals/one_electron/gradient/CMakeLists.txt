# The gradient launchers. Device code in its own library, so its language and -x
# hip never reach the module compiles. Its sources are the launcher and the
# generated kernels (kernels/), seven families of one unit per class, each
# compiled on its own: the launcher includes only the families' headers and
# launches through their tables. Under CUDA that wants relocatable device code,
# this library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns
# #20280 without it). The bridge header it shares with the module names int32_t,
# PrimitivePairClassView, NucleiView, wwrStream_t and wwrError_t, hence
# types_headers, mmd::headers, molecule::headers and wwr::device PUBLIC. The
# kernels include FelizJueves's felizjueves/runtime.h and (gv, ipv, iprv) the
# shared eval_boys.h, which nothing outside this library sees, hence
# felizjueves::headers and shared::headers PRIVATE; so is the launcher's block
# sizing (block_params.h, primitives.h), hence core::kernel_headers.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.gradient.device
  SOURCES
  gradient_launch.cu
  kernels/gs_00.cu
  kernels/gs_01.cu
  kernels/gs_02.cu
  kernels/gs_03.cu
  kernels/gs_11.cu
  kernels/gs_12.cu
  kernels/gs_13.cu
  kernels/gs_22.cu
  kernels/gs_23.cu
  kernels/gs_33.cu
  kernels/gt_00.cu
  kernels/gt_01.cu
  kernels/gt_02.cu
  kernels/gt_03.cu
  kernels/gt_11.cu
  kernels/gt_12.cu
  kernels/gt_13.cu
  kernels/gt_22.cu
  kernels/gt_23.cu
  kernels/gt_33.cu
  kernels/gv_00.cu
  kernels/gv_01.cu
  kernels/gv_02.cu
  kernels/gv_03.cu
  kernels/gv_11.cu
  kernels/gv_12.cu
  kernels/gv_13.cu
  kernels/gv_22.cu
  kernels/gv_23.cu
  kernels/gv_33.cu
  kernels/ips_00.cu
  kernels/ips_01.cu
  kernels/ips_02.cu
  kernels/ips_03.cu
  kernels/ips_11.cu
  kernels/ips_12.cu
  kernels/ips_13.cu
  kernels/ips_22.cu
  kernels/ips_23.cu
  kernels/ips_33.cu
  kernels/ipt_00.cu
  kernels/ipt_01.cu
  kernels/ipt_02.cu
  kernels/ipt_03.cu
  kernels/ipt_11.cu
  kernels/ipt_12.cu
  kernels/ipt_13.cu
  kernels/ipt_22.cu
  kernels/ipt_23.cu
  kernels/ipt_33.cu
  kernels/ipv_00.cu
  kernels/ipv_01.cu
  kernels/ipv_02.cu
  kernels/ipv_03.cu
  kernels/ipv_11.cu
  kernels/ipv_12.cu
  kernels/ipv_13.cu
  kernels/ipv_22.cu
  kernels/ipv_23.cu
  kernels/ipv_33.cu
  kernels/iprv_00.cu
  kernels/iprv_01.cu
  kernels/iprv_02.cu
  kernels/iprv_03.cu
  kernels/iprv_11.cu
  kernels/iprv_12.cu
  kernels/iprv_13.cu
  kernels/iprv_22.cu
  kernels/iprv_23.cu
  kernels/iprv_33.cu
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

# The module. DeviceResources, BasisSet, PrimitivePairClass, Molecule,
# SphericalMatrixAO and Buffers are in its signatures, so core::resources,
# basis, mmd, molecule, recipes::matrix_ao and core::wwr are PUBLIC. The device
# library is PRIVATE: the implementation unit includes the bridge header in its
# GMF and calls the launchers, and a static library's private link dependencies
# still reach the final link. one_electron::launch is PRIVATE too: only the
# implementation runs its checks and launch_each.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  gradient.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
  free_breakfast::core::wwr
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::one_electron::gradient::device
)
