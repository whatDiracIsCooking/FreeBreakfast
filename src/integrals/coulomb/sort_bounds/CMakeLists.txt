# The ket sort's device code: the key and gather kernels and their launcher.
# Device code in its own library, so its language and -x hip never reach the
# module compiles. The bridge header names int32_t, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, mmd::headers and
# wwr::device PUBLIC; the kernels read kWarpSize (block_params.h) and idivup
# (primitives.h), hence core::kernel_headers. The merge sort the launcher sorts
# with is src/integrals/sort's, PRIVATE: only the .cu includes its bridge.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.coulomb.sort_bounds.device
  SOURCES
  sort_bounds_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::core::kernel_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::integrals::sort::device
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
