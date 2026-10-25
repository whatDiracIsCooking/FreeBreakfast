# The stable merge sort of double keys the integral builds sort with: mmd
# orders each class's records by zeta, coulomb/sort_bounds each ket class by
# Iq * Dmax. Device code only, no module: a device library links it and its
# .cu includes merge_sort_bridge.h, which its source directory, its public
# include path, supplies. experimental/sort/merge links it too, rather than
# keeping a copy. The bridge names int32_t, wwrStream_t and wwrError_t, and
# merge_sort.cu reads kWarpSize (block_params.h) and math::idivup
# (primitives.h), hence core::kernel_headers PUBLIC: it brings
# primitives_header, types_headers and wwr::device with it.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.sort.device
  SOURCES
  merge_sort.cu
  LINK_PUBLIC
  free_breakfast::core::kernel_headers
)
