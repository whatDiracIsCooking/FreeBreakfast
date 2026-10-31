# The DF Hessian's device library: the generated kernels d2metric<lP, lQ> and
# d2eri3<la, lb, lP> (kernels/, written by tools/emit/df_hessian_kernels.py),
# one per class, the metric's lP <= lQ through i and the three-center's
# la <= lb through f against lP through i, each compiled on its own, and
# hessian_launch.cu, the per-class launchers, which check their tables against
# the energy kernels' limits at compile time. Device code in its own library,
# so its language and -x hip never reach a module compile. Under CUDA the
# launchers launch specializations other TUs define, which wants relocatable
# device code, this library's default, so it must never pass
# NO_DEVICE_LINKING (nvcc warns #20280 without it). The bridge header it shares
# with the module includes the metric's and the three-center's bridges for
# their limits and record layouts, hence their device libraries PUBLIC, beside
# types_headers and wwr::device for int32_t, wwrStream_t and wwrError_t. The
# kernels include FelizJueves's felizjueves/runtime.h (felizjueves::headers)
# and the shared eval_boys.h (shared::headers), and the launchers size their
# grids with kFourWarps and math::idivup (core::kernel_headers); all PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.hessian.device
  SOURCES
  hessian_launch.cu
  kernels/d2metric_00.cu
  kernels/d2metric_01.cu
  kernels/d2metric_02.cu
  kernels/d2metric_03.cu
  kernels/d2metric_04.cu
  kernels/d2metric_05.cu
  kernels/d2metric_06.cu
  kernels/d2metric_11.cu
  kernels/d2metric_12.cu
  kernels/d2metric_13.cu
  kernels/d2metric_14.cu
  kernels/d2metric_15.cu
  kernels/d2metric_16.cu
  kernels/d2metric_22.cu
  kernels/d2metric_23.cu
  kernels/d2metric_24.cu
  kernels/d2metric_25.cu
  kernels/d2metric_26.cu
  kernels/d2metric_33.cu
  kernels/d2metric_34.cu
  kernels/d2metric_35.cu
  kernels/d2metric_36.cu
  kernels/d2metric_44.cu
  kernels/d2metric_45.cu
  kernels/d2metric_46.cu
  kernels/d2metric_55.cu
  kernels/d2metric_56.cu
  kernels/d2metric_66.cu
  kernels/d2eri3_000.cu
  kernels/d2eri3_001.cu
  kernels/d2eri3_002.cu
  kernels/d2eri3_003.cu
  kernels/d2eri3_004.cu
  kernels/d2eri3_005.cu
  kernels/d2eri3_006.cu
  kernels/d2eri3_010.cu
  kernels/d2eri3_011.cu
  kernels/d2eri3_012.cu
  kernels/d2eri3_013.cu
  kernels/d2eri3_014.cu
  kernels/d2eri3_015.cu
  kernels/d2eri3_016.cu
  kernels/d2eri3_020.cu
  kernels/d2eri3_021.cu
  kernels/d2eri3_022.cu
  kernels/d2eri3_023.cu
  kernels/d2eri3_024.cu
  kernels/d2eri3_025.cu
  kernels/d2eri3_026.cu
  kernels/d2eri3_030.cu
  kernels/d2eri3_031.cu
  kernels/d2eri3_032.cu
  kernels/d2eri3_033.cu
  kernels/d2eri3_034.cu
  kernels/d2eri3_035.cu
  kernels/d2eri3_036.cu
  kernels/d2eri3_110.cu
  kernels/d2eri3_111.cu
  kernels/d2eri3_112.cu
  kernels/d2eri3_113.cu
  kernels/d2eri3_114.cu
  kernels/d2eri3_115.cu
  kernels/d2eri3_116.cu
  kernels/d2eri3_120.cu
  kernels/d2eri3_121.cu
  kernels/d2eri3_122.cu
  kernels/d2eri3_123.cu
  kernels/d2eri3_124.cu
  kernels/d2eri3_125.cu
  kernels/d2eri3_126.cu
  kernels/d2eri3_130.cu
  kernels/d2eri3_131.cu
  kernels/d2eri3_132.cu
  kernels/d2eri3_133.cu
  kernels/d2eri3_134.cu
  kernels/d2eri3_135.cu
  kernels/d2eri3_136.cu
  kernels/d2eri3_220.cu
  kernels/d2eri3_221.cu
  kernels/d2eri3_222.cu
  kernels/d2eri3_223.cu
  kernels/d2eri3_224.cu
  kernels/d2eri3_225.cu
  kernels/d2eri3_226.cu
  kernels/d2eri3_230.cu
  kernels/d2eri3_231.cu
  kernels/d2eri3_232.cu
  kernels/d2eri3_233.cu
  kernels/d2eri3_234.cu
  kernels/d2eri3_235.cu
  kernels/d2eri3_236.cu
  kernels/d2eri3_330.cu
  kernels/d2eri3_331.cu
  kernels/d2eri3_332.cu
  kernels/d2eri3_333.cu
  kernels/d2eri3_334.cu
  kernels/d2eri3_335.cu
  kernels/d2eri3_336.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::eri::density_fitting::metric::device
  free_breakfast::integrals::eri::density_fitting::three_center::device
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)

# The module. DeviceResources, BasisSet, PrimitivePairClass and Result are in
# its signature, so core::resources, core::error_handling, basis and mmd are
# PUBLIC. The device library is PRIVATE: the implementation unit includes the
# bridge header in its GMF and calls the launchers, and a static library's
# private link dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  IMPLEMENTATION
  hessian.cpp
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
  free_breakfast::integrals::eri::density_fitting::hessian::device
)
