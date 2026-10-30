# The J gradient's launchers: the gather of each record's two atoms and gjab.
# Device code in its own library, so its language and -x hip never reach the
# module compiles. Its sources are the launcher and the generated gjab kernels
# (kernels/, written by tools/emit/coulomb_kernels.py), one per class, each
# compiled on its own: the launcher includes only kernels/gjab.h and launches
# through its table. Under CUDA that wants relocatable device code, this
# library's default, so it must never pass NO_DEVICE_LINKING (nvcc warns #20280
# without it). The bridge header it shares with the module names int32_t,
# PrimitivePairClassView, wwrStream_t and wwrError_t, hence types_headers,
# mmd::headers and wwr::device PUBLIC. The kernels include FelizJueves's
# felizjueves/runtime.h (felizjueves::headers), the launcher's block sizing
# reads kFourWarps and idivup (block_params.h, primitives.h), hence
# core::kernel_headers, and it checks its table against J's limit,
# kCoulombMaxMomentum (pair_class's bridge), hence pair_class::device; all three
# PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb.gradient.device
  SOURCES
  coulomb_gradient_launch.cu
  kernels/gjab_00.cu
  kernels/gjab_01.cu
  kernels/gjab_02.cu
  kernels/gjab_03.cu
  kernels/gjab_11.cu
  kernels/gjab_12.cu
  kernels/gjab_13.cu
  kernels/gjab_22.cu
  kernels/gjab_23.cu
  kernels/gjab_33.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class::device
)
