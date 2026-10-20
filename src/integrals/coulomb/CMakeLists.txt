# The J build's device code. free_breakfast.integrals.coulomb has no launcher
# or module yet, only the generated kernels (kernels/, written by
# tools/emit_coulomb_kernels.py) and coulomb_kernels.cu, which asserts at
# compile time that their tables were emitted for kMaxAngularMomentum and that
# e_hermite's block matches the build's warp width. Each kernel is its own
# source, one per class, compiled on its own: relocatable device code, this
# library's default, is what a launcher in another TU will need to launch
# through the tables (nvcc warns #20280 without it), so it must never pass
# NO_DEVICE_LINKING. The units include FelizJueves's felizjueves/runtime.h
# (felizjueves::headers, which also carries FJ_WAVEFRONT = WWR_WARP_SIZE);
# coulomb_kernels.cu reads kMaxAngularMomentum (basis::headers) and kWarpSize
# (core::kernel_headers).
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.coulomb.device
  SOURCES
  coulomb_kernels.cu
  kernels/dq/dq_00.cu
  kernels/dq/dq_01.cu
  kernels/dq/dq_02.cu
  kernels/dq/dq_11.cu
  kernels/dq/dq_12.cu
  kernels/dq/dq_22.cu
  kernels/e_hermite/e_hermite_00.cu
  kernels/e_hermite/e_hermite_01.cu
  kernels/e_hermite/e_hermite_02.cu
  kernels/e_hermite/e_hermite_03.cu
  kernels/e_hermite/e_hermite_04.cu
  kernels/e_hermite/e_hermite_10.cu
  kernels/e_hermite/e_hermite_11.cu
  kernels/e_hermite/e_hermite_12.cu
  kernels/e_hermite/e_hermite_13.cu
  kernels/e_hermite/e_hermite_14.cu
  kernels/e_hermite/e_hermite_20.cu
  kernels/e_hermite/e_hermite_21.cu
  kernels/e_hermite/e_hermite_22.cu
  kernels/e_hermite/e_hermite_23.cu
  kernels/e_hermite/e_hermite_24.cu
  kernels/e_hermite/e_hermite_30.cu
  kernels/e_hermite/e_hermite_31.cu
  kernels/e_hermite/e_hermite_32.cu
  kernels/e_hermite/e_hermite_33.cu
  kernels/e_hermite/e_hermite_34.cu
  kernels/e_hermite/e_hermite_40.cu
  kernels/e_hermite/e_hermite_41.cu
  kernels/e_hermite/e_hermite_42.cu
  kernels/e_hermite/e_hermite_43.cu
  kernels/e_hermite/e_hermite_44.cu
  kernels/jab/jab_00.cu
  kernels/jab/jab_01.cu
  kernels/jab/jab_02.cu
  kernels/jab/jab_11.cu
  kernels/jab/jab_12.cu
  kernels/jab/jab_22.cu
  LINK_PRIVATE
  free_breakfast::basis::headers
  free_breakfast::core::kernel_headers
  felizjueves::headers
)
