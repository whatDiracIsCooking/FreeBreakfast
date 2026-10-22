# The exchange build's hand-written device steps -- max|D|, the Dcd gather, the
# block table, the shell starts and the mirror -- as plain-pointer launchers in
# a device library of their
# own, so their language and -x hip never reach the module compiles. Nothing
# here launches another TU's kernel, so the library needs no relocatable
# device code, but keeps the default as coulomb's does. The bridge header names
# int32_t, wwrStream_t and wwrError_t, hence types_headers and wwr::device
# PUBLIC; the .cu reads kMaxAngularMomentum (angular_momentum.h), and
# block_reduce, kWarpSize and math::fabs / idivup (block_reduce.cuh,
# block_params.h, primitives.h), hence basis::headers and core::kernel_headers
# PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.exchange.helpers
  SOURCES
  exchange_helpers.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::basis::headers
  free_breakfast::core::kernel_headers
)
