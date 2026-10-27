# The exchange build's hand-written device steps -- max|D|, the Dcd gather, the
# block table, the shell starts, the mirror and the rolled kernel's E tables --
# as plain-pointer launchers in a device library of their own, so their
# language and -x hip never reach the module compiles. One .cu per step, the
# kernels that share private machinery together; what crosses files is
# exchange_helpers.cuh. Nothing here launches another TU's kernel, so the
# library needs no relocatable device code, but keeps the default as coulomb's
# does. The bridge header names int32_t,
# wwrStream_t and wwrError_t, hence types_headers and wwr::device PUBLIC; the
# .cu files read block_reduce, kWarpSize, kMaxThreads and math::fabs / exp /
# idivup (block_reduce.cuh, block_params.h, primitives.h), hence
# core::kernel_headers PRIVATE. Their L limit is the bridge's
# kExchangeMaxMomentum, not angular_momentum.h's.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.exchange.helpers
  SOURCES
  blocks.cu
  density.cu
  e_table.cu
  gather_dcd.cu
  geometry.cu
  mirror.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)
