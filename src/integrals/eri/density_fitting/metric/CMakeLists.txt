# The DF Coulomb metric's device library: the generated kernels metric<lP, lQ>
# (kernels/, written by tools/emit/df_kernels.py), one per aux class lP <= lQ
# through g, each compiled on its own, and metric_launch.cu, the per-class
# launcher, which checks their table (kernels/metric.h) against the limit in
# metric_bridge.h at compile time. Device code in its own library, so its
# language and -x hip never reach a module compile. Under CUDA the launcher
# launches specializations other TUs define, which wants relocatable device
# code, this library's default, so it must never pass NO_DEVICE_LINKING (nvcc
# warns #20280 without it). The bridge header it shares with the module names
# int32_t, wwrStream_t and wwrError_t, hence types_headers and wwr::device
# PUBLIC. The kernels include FelizJueves's felizjueves/runtime.h
# (felizjueves::headers) and the shared eval_boys.h (shared::headers), and the
# launcher sizes its grid with kFourWarps and math::idivup (block_params.h,
# primitives.h, core::kernel_headers); all PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.metric.device
  SOURCES
  metric_launch.cu
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
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)

# The module. DeviceResources, BasisSet and MatrixAOPair are in its signature,
# so core::resources, basis and recipes::matrix_ao are PUBLIC. The device
# library is PRIVATE: the implementation unit includes the bridge header in
# its GMF and calls the launcher, and a static library's private link
# dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.metric
  PRIMARY_INTERFACE
  metric.cppm
  IMPLEMENTATION
  metric.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::density_fitting::metric::device
)
