# The J launchers. Device code in its own library, so its language and -x hip
# never reach the module compiles. Its sources are the launcher and the
# generated kernels (kernels/, written by tools/emit_coulomb_kernels.py), one
# per class, each compiled on its own: the launcher includes only the three
# families' headers and launches through their tables. Under CUDA that wants
# relocatable device code, this library's default, so it must never pass
# NO_DEVICE_LINKING (nvcc warns #20280 without it). The bridge header it shares
# with the module names int32_t, kMaxAngularMomentum, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, basis::headers,
# mmd::headers and wwr::device PUBLIC. The kernels include FelizJueves's
# felizjueves/runtime.h (felizjueves::headers, which also carries FJ_WAVEFRONT
# = WWR_WARP_SIZE), and the launcher's block sizing reads kWarpSize
# (block_params.h, primitives.h), hence core::kernel_headers; both PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.coulomb.device
  SOURCES
  coulomb_launch.cu
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
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
)

# The module. DeviceResources, PrimitivePairClass and CartesianMatrixAO are in
# its signature, so core::resources, mmd and linalg::matrix_ao are PUBLIC. The
# per-class launch and the pre-launch checks are the one-electron builds' own
# (one_electron::launch), and the device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launchers, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.coulomb
  PRIMARY_INTERFACE
  coulomb.cppm
  IMPLEMENTATION
  coulomb.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::mmd
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::coulomb::device
)
