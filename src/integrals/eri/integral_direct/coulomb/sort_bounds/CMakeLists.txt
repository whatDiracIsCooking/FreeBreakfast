# The ket sort's device code: the key and gather kernels and their launcher,
# driven through its bridge header by coulomb's implementation unit (which
# owns the buffers, in CoulombPairClass) and by test/gtest/sort_bounds_gpu.cpp.
# Device code in its own library, so its language and -x hip never reach the
# module compiles. The bridge header names int32_t, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, mmd::headers and
# wwr::device PUBLIC; the kernels read kWarpSize (block_params.h) and idivup
# (primitives.h), hence core::kernel_headers. The merge sort the launcher sorts
# with is src/recipes/sort's, PRIVATE: only the .cu includes its bridge.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.sort_bounds.device
  SOURCES
  sort_bounds_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::core::kernel_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::recipes::sort::device
)
