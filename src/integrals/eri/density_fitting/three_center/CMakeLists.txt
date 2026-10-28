# The DF three-center integrals' device library: the generated kernels
# eri3<la, lb, lP> (kernels/, written by tools/emit/df_kernels.py), one per
# class la <= lb through f against lP through i, each compiled on its own, and
# three_center_launch.cu, the per-class launcher and the (m, n)
# symmetrization, which checks their table (kernels/eri3.h) against the limits
# in three_center_bridge.h at compile time. Device code in its own library, so
# its language and -x hip never reach a module compile. Under CUDA the launcher
# launches specializations other TUs define, which wants relocatable device
# code, this library's default, so it must never pass NO_DEVICE_LINKING (nvcc
# warns #20280 without it). The bridge header it shares with the module names
# int32_t, kMaxAngularMomentum, wwrStream_t and wwrError_t, hence
# types_headers, basis::headers and wwr::device PUBLIC. The kernels include
# FelizJueves's felizjueves/runtime.h (felizjueves::headers) and the shared
# eval_boys.h (shared::headers), and the launcher sizes its grids with
# kFourWarps and math::idivup (core::kernel_headers) and symmetrizes with kHalf
# (core::constants_header); all PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.three_center.device
  SOURCES
  three_center_launch.cu
  kernels/eri3_000.cu
  kernels/eri3_001.cu
  kernels/eri3_002.cu
  kernels/eri3_003.cu
  kernels/eri3_004.cu
  kernels/eri3_005.cu
  kernels/eri3_006.cu
  kernels/eri3_010.cu
  kernels/eri3_011.cu
  kernels/eri3_012.cu
  kernels/eri3_013.cu
  kernels/eri3_014.cu
  kernels/eri3_015.cu
  kernels/eri3_016.cu
  kernels/eri3_020.cu
  kernels/eri3_021.cu
  kernels/eri3_022.cu
  kernels/eri3_023.cu
  kernels/eri3_024.cu
  kernels/eri3_025.cu
  kernels/eri3_026.cu
  kernels/eri3_030.cu
  kernels/eri3_031.cu
  kernels/eri3_032.cu
  kernels/eri3_033.cu
  kernels/eri3_034.cu
  kernels/eri3_035.cu
  kernels/eri3_036.cu
  kernels/eri3_110.cu
  kernels/eri3_111.cu
  kernels/eri3_112.cu
  kernels/eri3_113.cu
  kernels/eri3_114.cu
  kernels/eri3_115.cu
  kernels/eri3_116.cu
  kernels/eri3_120.cu
  kernels/eri3_121.cu
  kernels/eri3_122.cu
  kernels/eri3_123.cu
  kernels/eri3_124.cu
  kernels/eri3_125.cu
  kernels/eri3_126.cu
  kernels/eri3_130.cu
  kernels/eri3_131.cu
  kernels/eri3_132.cu
  kernels/eri3_133.cu
  kernels/eri3_134.cu
  kernels/eri3_135.cu
  kernels/eri3_136.cu
  kernels/eri3_220.cu
  kernels/eri3_221.cu
  kernels/eri3_222.cu
  kernels/eri3_223.cu
  kernels/eri3_224.cu
  kernels/eri3_225.cu
  kernels/eri3_226.cu
  kernels/eri3_230.cu
  kernels/eri3_231.cu
  kernels/eri3_232.cu
  kernels/eri3_233.cu
  kernels/eri3_234.cu
  kernels/eri3_235.cu
  kernels/eri3_236.cu
  kernels/eri3_330.cu
  kernels/eri3_331.cu
  kernels/eri3_332.cu
  kernels/eri3_333.cu
  kernels/eri3_334.cu
  kernels/eri3_335.cu
  kernels/eri3_336.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)

# The module. DeviceResources, BasisSet and PrimitivePairClass are in its
# signature, so core::resources, basis and mmd are PUBLIC. The device library
# is PRIVATE: the implementation unit includes the bridge header in its GMF and
# calls the launchers, and a static library's private link dependencies still
# reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.three_center
  PRIMARY_INTERFACE
  three_center.cppm
  IMPLEMENTATION
  three_center.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::eri::density_fitting::three_center::device
)
