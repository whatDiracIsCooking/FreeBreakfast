# The [p|q] contraction's launcher. Device code in its own library, so its
# language and -x hip never reach the module compiles. Its sources are the
# launcher and the generated kernels (kernels/, written by
# tools/emit/coulomb_kernels.py), one per Hermite class, each compiled on its
# own: the launcher includes only kernels/e_hermite.h and launches through its
# table. Under CUDA that wants relocatable device code, this library's
# default, so it must never pass NO_DEVICE_LINKING (nvcc warns #20280 without
# it). The bridge header it shares with the module names int32_t,
# PrimitivePairClassView, wwrStream_t and wwrError_t, hence types_headers,
# mmd::headers and wwr::device PUBLIC. The kernels include FelizJueves's
# felizjueves/runtime.h (felizjueves::headers, which also carries FJ_WAVEFRONT
# = WWR_WARP_SIZE) and the shared eval_boys.h (shared::headers), and the
# launcher checks e_hermite's block against kWarpSize (block_params.h), hence
# core::kernel_headers, and its table against J's limit, kCoulombMaxMomentum
# (pair_class's bridge), hence pair_class::device; all four PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.e_hermite.device
  SOURCES
  e_hermite_launch.cu
  kernels/e_hermite_00.cu
  kernels/e_hermite_01.cu
  kernels/e_hermite_02.cu
  kernels/e_hermite_03.cu
  kernels/e_hermite_04.cu
  kernels/e_hermite_05.cu
  kernels/e_hermite_06.cu
  kernels/e_hermite_10.cu
  kernels/e_hermite_11.cu
  kernels/e_hermite_12.cu
  kernels/e_hermite_13.cu
  kernels/e_hermite_14.cu
  kernels/e_hermite_15.cu
  kernels/e_hermite_16.cu
  kernels/e_hermite_20.cu
  kernels/e_hermite_21.cu
  kernels/e_hermite_22.cu
  kernels/e_hermite_23.cu
  kernels/e_hermite_24.cu
  kernels/e_hermite_25.cu
  kernels/e_hermite_26.cu
  kernels/e_hermite_30.cu
  kernels/e_hermite_31.cu
  kernels/e_hermite_32.cu
  kernels/e_hermite_33.cu
  kernels/e_hermite_34.cu
  kernels/e_hermite_35.cu
  kernels/e_hermite_36.cu
  kernels/e_hermite_40.cu
  kernels/e_hermite_41.cu
  kernels/e_hermite_42.cu
  kernels/e_hermite_43.cu
  kernels/e_hermite_44.cu
  kernels/e_hermite_45.cu
  kernels/e_hermite_46.cu
  kernels/e_hermite_50.cu
  kernels/e_hermite_51.cu
  kernels/e_hermite_52.cu
  kernels/e_hermite_53.cu
  kernels/e_hermite_54.cu
  kernels/e_hermite_55.cu
  kernels/e_hermite_56.cu
  kernels/e_hermite_60.cu
  kernels/e_hermite_61.cu
  kernels/e_hermite_62.cu
  kernels/e_hermite_63.cu
  kernels/e_hermite_64.cu
  kernels/e_hermite_65.cu
  kernels/e_hermite_66.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class::device
  free_breakfast::integrals::shared::headers
)

# The module. CoulombPairClass and wwrError_t are in its signature, so
# pair_class and core::wwr are PUBLIC; the implementation unit reads the
# classes' PrimitivePairClassView (mmd). The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.e_hermite
  PRIMARY_INTERFACE
  e_hermite.cppm
  IMPLEMENTATION
  e_hermite.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::wwr
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class
  LINK_PRIVATE
  free_breakfast::integrals::mmd
  free_breakfast::integrals::eri::integral_direct::coulomb::e_hermite::device
)
