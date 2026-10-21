# The ket sort's device code: the key and gather kernels and their launcher,
# and the merge sort they sort with. Device code in its own library, so its
# language and -x hip never reach the module compiles. Its source directory is
# its public include path, so merge_sort_bridge.h reaches
# experimental/sort/merge too, which links this rather than keeping a copy of
# merge_sort.cu. The bridge headers name int32_t, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, mmd::headers and
# wwr::device PUBLIC; the kernels read kWarpSize (block_params.h) and idivup
# (primitives.h), hence core::kernel_headers, PUBLIC as merge_sort.cu's
# experimental library had it.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.coulomb.sort_bounds.device
  SOURCES
  sort_bounds_launch.cu
  merge_sort.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::core::kernel_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
)

# The module. Device, Buffers and PrimitivePairClass are in its interface, so
# core::wwr and mmd are PUBLIC. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.coulomb.sort_bounds
  PRIMARY_INTERFACE
  sort_bounds.cppm
  IMPLEMENTATION
  sort_bounds.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::integrals::mmd
  LINK_PRIVATE
  free_breakfast::integrals::coulomb::sort_bounds::device
)
