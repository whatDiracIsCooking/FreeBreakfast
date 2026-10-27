# The stable merge sort of double keys, and the column gather that applies its
# permutation. Device code in its own library, so its language and -x hip never
# reach the module compiles. The integral builds' device libraries link it
# directly and their .cu include merge_sort_bridge.h, which its source
# directory, its public include path, supplies: mmd orders each class's records
# by zeta, coulomb/sort_bounds each ket class by Iq * Dmax.
# experimental/sort/merge links the module, rather than keeping a copy. The
# bridges name int32_t, wwrStream_t and wwrError_t, and the kernels read
# kWarpSize and kFourWarps (block_params.h) and math::idivup (primitives.h),
# hence core::kernel_headers PUBLIC: it brings primitives_header, types_headers
# and wwr::device with it.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.sort.device
  SOURCES
  merge_sort.cu
  gather_columns.cu
  LINK_PUBLIC
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources and Status are in its signatures, so
# core::resources and core::error_handling are PUBLIC. The device library is
# PRIVATE: the implementation unit includes the bridge headers in its GMF and
# calls the launchers. core::primitives (align_up) and core::wwr
# (ScopedDeviceIndex, gpu_status) are its own business too.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.sort
  PRIMARY_INTERFACE
  sort.cppm
  IMPLEMENTATION
  sort.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::recipes::sort::device
)
