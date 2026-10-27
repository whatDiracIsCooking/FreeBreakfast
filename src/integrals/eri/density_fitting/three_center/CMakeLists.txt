# The DF three-center integrals' generated kernels, eri3<la, lb, lP> (kernels/,
# written by tools/emit_df_kernels.py), one per class la <= lb through f
# against lP through g, each compiled on its own, and three_center_table.cu,
# which checks their table (kernels/eri3.h) against the limits in
# three_center_bridge.h at compile time. No launcher yet. Device code in its own
# library, so its language and -x hip never reach a module compile. Under CUDA
# a launcher in another TU of this library will want relocatable device code,
# this library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns
# #20280 without it). The bridge header names int32_t and kMaxAngularMomentum,
# hence types_headers and basis::headers PUBLIC. The kernels include
# FelizJueves's felizjueves/runtime.h (felizjueves::headers) and the shared
# eval_boys.h (shared::headers); both PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.three_center.device
  SOURCES
  three_center_table.cu
  kernels/eri3_000.cu
  kernels/eri3_001.cu
  kernels/eri3_002.cu
  kernels/eri3_003.cu
  kernels/eri3_004.cu
  kernels/eri3_010.cu
  kernels/eri3_011.cu
  kernels/eri3_012.cu
  kernels/eri3_013.cu
  kernels/eri3_014.cu
  kernels/eri3_020.cu
  kernels/eri3_021.cu
  kernels/eri3_022.cu
  kernels/eri3_023.cu
  kernels/eri3_024.cu
  kernels/eri3_030.cu
  kernels/eri3_031.cu
  kernels/eri3_032.cu
  kernels/eri3_033.cu
  kernels/eri3_034.cu
  kernels/eri3_110.cu
  kernels/eri3_111.cu
  kernels/eri3_112.cu
  kernels/eri3_113.cu
  kernels/eri3_114.cu
  kernels/eri3_120.cu
  kernels/eri3_121.cu
  kernels/eri3_122.cu
  kernels/eri3_123.cu
  kernels/eri3_124.cu
  kernels/eri3_130.cu
  kernels/eri3_131.cu
  kernels/eri3_132.cu
  kernels/eri3_133.cu
  kernels/eri3_134.cu
  kernels/eri3_220.cu
  kernels/eri3_221.cu
  kernels/eri3_222.cu
  kernels/eri3_223.cu
  kernels/eri3_224.cu
  kernels/eri3_230.cu
  kernels/eri3_231.cu
  kernels/eri3_232.cu
  kernels/eri3_233.cu
  kernels/eri3_234.cu
  kernels/eri3_330.cu
  kernels/eri3_331.cu
  kernels/eri3_332.cu
  kernels/eri3_333.cu
  kernels/eri3_334.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  LINK_PRIVATE
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)
