# The exchange build's hand-written device steps (helpers/).
add_subdirectory(helpers)

# The ordered pair classes K's bras and kets are drawn from (pairs/), before
# the device library below: its bridge header names their view.
add_subdirectory(pairs)

# The K launchers. Device code in its own library, so its language and -x hip
# never reach a module compile. Its sources are the launcher and the generated
# kernels k<la, lc, lb, ld> (kernels/k/, written by
# tools/emit_exchange_kernels.py), one per K class through d, 81 in all, each
# compiled on its own: the launcher includes only kernels/k/k.h and launches
# through its table. Under CUDA that wants relocatable device code, this
# library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns #20280
# without it). The bridge header it shares with the module names int32_t,
# OrderedPairClassView, wwrStream_t and wwrError_t, hence types_headers,
# pairs::headers and wwr::device PUBLIC. The kernels include FelizJueves's
# felizjueves/runtime.h (felizjueves::headers, which also carries FJ_WAVEFRONT
# = WWR_WARP_SIZE) and the shared eval_boys.h (shared::headers), and the
# launcher reads kMaxAngularMomentum (angular_momentum.h) and kWarpSize and
# math::idivup (block_params.h, primitives.h), hence basis::headers and
# core::kernel_headers; all four PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.exchange.device
  SOURCES
  exchange_launch.cu
  kernels/k/k_00_00.cu
  kernels/k/k_00_01.cu
  kernels/k/k_00_02.cu
  kernels/k/k_00_10.cu
  kernels/k/k_00_11.cu
  kernels/k/k_00_12.cu
  kernels/k/k_00_20.cu
  kernels/k/k_00_21.cu
  kernels/k/k_00_22.cu
  kernels/k/k_01_00.cu
  kernels/k/k_01_01.cu
  kernels/k/k_01_02.cu
  kernels/k/k_01_10.cu
  kernels/k/k_01_11.cu
  kernels/k/k_01_12.cu
  kernels/k/k_01_20.cu
  kernels/k/k_01_21.cu
  kernels/k/k_01_22.cu
  kernels/k/k_02_00.cu
  kernels/k/k_02_01.cu
  kernels/k/k_02_02.cu
  kernels/k/k_02_10.cu
  kernels/k/k_02_11.cu
  kernels/k/k_02_12.cu
  kernels/k/k_02_20.cu
  kernels/k/k_02_21.cu
  kernels/k/k_02_22.cu
  kernels/k/k_10_00.cu
  kernels/k/k_10_01.cu
  kernels/k/k_10_02.cu
  kernels/k/k_10_10.cu
  kernels/k/k_10_11.cu
  kernels/k/k_10_12.cu
  kernels/k/k_10_20.cu
  kernels/k/k_10_21.cu
  kernels/k/k_10_22.cu
  kernels/k/k_11_00.cu
  kernels/k/k_11_01.cu
  kernels/k/k_11_02.cu
  kernels/k/k_11_10.cu
  kernels/k/k_11_11.cu
  kernels/k/k_11_12.cu
  kernels/k/k_11_20.cu
  kernels/k/k_11_21.cu
  kernels/k/k_11_22.cu
  kernels/k/k_12_00.cu
  kernels/k/k_12_01.cu
  kernels/k/k_12_02.cu
  kernels/k/k_12_10.cu
  kernels/k/k_12_11.cu
  kernels/k/k_12_12.cu
  kernels/k/k_12_20.cu
  kernels/k/k_12_21.cu
  kernels/k/k_12_22.cu
  kernels/k/k_20_00.cu
  kernels/k/k_20_01.cu
  kernels/k/k_20_02.cu
  kernels/k/k_20_10.cu
  kernels/k/k_20_11.cu
  kernels/k/k_20_12.cu
  kernels/k/k_20_20.cu
  kernels/k/k_20_21.cu
  kernels/k/k_20_22.cu
  kernels/k/k_21_00.cu
  kernels/k/k_21_01.cu
  kernels/k/k_21_02.cu
  kernels/k/k_21_10.cu
  kernels/k/k_21_11.cu
  kernels/k/k_21_12.cu
  kernels/k/k_21_20.cu
  kernels/k/k_21_21.cu
  kernels/k/k_21_22.cu
  kernels/k/k_22_00.cu
  kernels/k/k_22_01.cu
  kernels/k/k_22_02.cu
  kernels/k/k_22_10.cu
  kernels/k/k_22_11.cu
  kernels/k/k_22_12.cu
  kernels/k/k_22_20.cu
  kernels/k/k_22_21.cu
  kernels/k/k_22_22.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::exchange::pairs::headers
  wwr::device
  LINK_PRIVATE
  felizjueves::headers
  free_breakfast::integrals::shared::headers
  free_breakfast::basis::headers
  free_breakfast::core::kernel_headers
)

# The module. DeviceResources, OrderedPairClass and CartesianMatrixAO are in its
# signature, so core::resources, exchange::pairs (re-exported) and
# linalg::matrix_ao are PUBLIC. The per-class launch is the one-electron
# builds' (one_electron::launch), and the two device libraries are PRIVATE: the
# implementation unit includes their bridge headers in its GMF and calls the
# launchers, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.exchange
  PRIMARY_INTERFACE
  exchange.cppm
  IMPLEMENTATION
  exchange.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::exchange::pairs
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::basis
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::exchange::device
  free_breakfast::integrals::exchange::helpers
)
