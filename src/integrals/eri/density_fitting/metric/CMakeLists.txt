# The DF Coulomb metric's generated kernels, metric<lP, lQ> (kernels/, written
# by tools/emit_df_kernels.py), one per aux class lP <= lQ through g, each
# compiled on its own, and metric_table.cu, which checks their table
# (kernels/metric.h) against the limit in metric_bridge.h at compile time. No
# launcher yet. Device code in its own library, so its language and -x hip
# never reach a module compile. Under CUDA a launcher in another TU of this
# library will want relocatable device code, this library's default, so it
# must never pass NO_DEVICE_LINKING (nvcc warns #20280 without it). The bridge
# header names int32_t, hence types_headers PUBLIC. The kernels include
# FelizJueves's felizjueves/runtime.h (felizjueves::headers) and the shared
# eval_boys.h (shared::headers); both PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.metric.device
  SOURCES
  metric_table.cu
  kernels/metric_00.cu
  kernels/metric_01.cu
  kernels/metric_02.cu
  kernels/metric_03.cu
  kernels/metric_04.cu
  kernels/metric_11.cu
  kernels/metric_12.cu
  kernels/metric_13.cu
  kernels/metric_14.cu
  kernels/metric_22.cu
  kernels/metric_23.cu
  kernels/metric_24.cu
  kernels/metric_33.cu
  kernels/metric_34.cu
  kernels/metric_44.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  LINK_PRIVATE
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)
