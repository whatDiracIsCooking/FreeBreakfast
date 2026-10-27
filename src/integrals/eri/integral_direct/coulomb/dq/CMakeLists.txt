# The ket pre-contraction's launcher. Device code in its own library, so its
# language and -x hip never reach the module compiles. Its sources are the
# launcher and the generated kernels (kernels/, written by
# tools/emit_coulomb_kernels.py), one per class, each compiled on its own: the
# launcher includes only kernels/dq.h and launches through its table. Under
# CUDA that wants relocatable device code, this library's default, so it must
# never pass NO_DEVICE_LINKING (nvcc warns #20280 without it). The bridge
# header it shares with the module names int32_t, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, mmd::headers and wwr::device
# PUBLIC. The kernels include FelizJueves's felizjueves/runtime.h
# (felizjueves::headers), the launcher's block sizing reads kFourWarps and
# idivup (block_params.h, primitives.h), hence core::kernel_headers, and it
# checks its table against J's limit, kCoulombMaxMomentum (pair_class's bridge),
# hence pair_class::device; all three PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.dq.device
  SOURCES
  dq_launch.cu
  kernels/dq_00.cu
  kernels/dq_01.cu
  kernels/dq_02.cu
  kernels/dq_03.cu
  kernels/dq_11.cu
  kernels/dq_12.cu
  kernels/dq_13.cu
  kernels/dq_22.cu
  kernels/dq_23.cu
  kernels/dq_33.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class::device
)

# The module. CoulombPairClass, CartesianMatrixAO and wwrError_t are in its
# signature, so pair_class, recipes::matrix_ao and core::wwr are PUBLIC. The
# device library is PRIVATE: the implementation unit includes the bridge header
# in its GMF and calls the launcher, and a static library's private link
# dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.dq
  PRIMARY_INTERFACE
  dq.cppm
  IMPLEMENTATION
  dq.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::wwr
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::integrals::eri::integral_direct::coulomb::dq::device
)
