# The DF gradient's device library: the generated kernels gmetric<lP, lQ> and
# geri3<la, lb, lP> (kernels/, written by tools/emit/df_gradient_kernels.py),
# one per class, the metric's lP <= lQ through i and the three-center's
# la <= lb through f against lP through i, each compiled on its own, and
# gradient_launch.cu, the per-class launchers, which check their tables against
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
  free_breakfast.integrals.eri.density_fitting.gradient.device
  SOURCES
  gradient_launch.cu
  kernels/gmetric_00.cu
  kernels/gmetric_01.cu
  kernels/gmetric_02.cu
  kernels/gmetric_03.cu
  kernels/gmetric_04.cu
  kernels/gmetric_05.cu
  kernels/gmetric_06.cu
  kernels/gmetric_11.cu
  kernels/gmetric_12.cu
  kernels/gmetric_13.cu
  kernels/gmetric_14.cu
  kernels/gmetric_15.cu
  kernels/gmetric_16.cu
  kernels/gmetric_22.cu
  kernels/gmetric_23.cu
  kernels/gmetric_24.cu
  kernels/gmetric_25.cu
  kernels/gmetric_26.cu
  kernels/gmetric_33.cu
  kernels/gmetric_34.cu
  kernels/gmetric_35.cu
  kernels/gmetric_36.cu
  kernels/gmetric_44.cu
  kernels/gmetric_45.cu
  kernels/gmetric_46.cu
  kernels/gmetric_55.cu
  kernels/gmetric_56.cu
  kernels/gmetric_66.cu
  kernels/geri3_000.cu
  kernels/geri3_001.cu
  kernels/geri3_002.cu
  kernels/geri3_003.cu
  kernels/geri3_004.cu
  kernels/geri3_005.cu
  kernels/geri3_006.cu
  kernels/geri3_010.cu
  kernels/geri3_011.cu
  kernels/geri3_012.cu
  kernels/geri3_013.cu
  kernels/geri3_014.cu
  kernels/geri3_015.cu
  kernels/geri3_016.cu
  kernels/geri3_020.cu
  kernels/geri3_021.cu
  kernels/geri3_022.cu
  kernels/geri3_023.cu
  kernels/geri3_024.cu
  kernels/geri3_025.cu
  kernels/geri3_026.cu
  kernels/geri3_030.cu
  kernels/geri3_031.cu
  kernels/geri3_032.cu
  kernels/geri3_033.cu
  kernels/geri3_034.cu
  kernels/geri3_035.cu
  kernels/geri3_036.cu
  kernels/geri3_110.cu
  kernels/geri3_111.cu
  kernels/geri3_112.cu
  kernels/geri3_113.cu
  kernels/geri3_114.cu
  kernels/geri3_115.cu
  kernels/geri3_116.cu
  kernels/geri3_120.cu
  kernels/geri3_121.cu
  kernels/geri3_122.cu
  kernels/geri3_123.cu
  kernels/geri3_124.cu
  kernels/geri3_125.cu
  kernels/geri3_126.cu
  kernels/geri3_130.cu
  kernels/geri3_131.cu
  kernels/geri3_132.cu
  kernels/geri3_133.cu
  kernels/geri3_134.cu
  kernels/geri3_135.cu
  kernels/geri3_136.cu
  kernels/geri3_220.cu
  kernels/geri3_221.cu
  kernels/geri3_222.cu
  kernels/geri3_223.cu
  kernels/geri3_224.cu
  kernels/geri3_225.cu
  kernels/geri3_226.cu
  kernels/geri3_230.cu
  kernels/geri3_231.cu
  kernels/geri3_232.cu
  kernels/geri3_233.cu
  kernels/geri3_234.cu
  kernels/geri3_235.cu
  kernels/geri3_236.cu
  kernels/geri3_330.cu
  kernels/geri3_331.cu
  kernels/geri3_332.cu
  kernels/geri3_333.cu
  kernels/geri3_334.cu
  kernels/geri3_335.cu
  kernels/geri3_336.cu
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
  free_breakfast.integrals.eri.density_fitting.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  gradient.cpp
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
  free_breakfast::integrals::eri::density_fitting::gradient::device
)
