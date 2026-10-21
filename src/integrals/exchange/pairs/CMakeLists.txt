# OrderedPairClassView, the exchange launchers' view of an ordered class,
# offered two ways from one definition in ordered_pair_class_view.h: `#include`
# through this include path, and `import
# free_breakfast.integrals.exchange.pairs;`, which includes it in its GMF and
# re-exports the name (mmd's headers library, the same arrangement).
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.exchange.pairs.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# The record bound and order: each record's dressed Schwarz bound through the
# generated record_bound kernels (kernels/record_bound/, written by
# tools/emit_exchange_kernels.py), one per ordered class, each compiled on its
# own, and each group sorted by it, once per basis. Device code in its own
# library, so its language and -x hip never reach the module compiles. Under
# CUDA launching another unit's specialization wants relocatable device code,
# this library's default, so it must never pass NO_DEVICE_LINKING. The bridge
# header names int32_t, OrderedPairClassView, wwrStream_t and wwrError_t, hence
# types_headers, headers and wwr::device PUBLIC; the launcher reads
# kFourWarps and idivup (core::kernel_headers) and kMaxAngularMomentum
# (basis::headers), sorts with src/integrals/sort's merge sort, and the
# generated units include felizjueves/runtime.h (felizjueves::headers); all
# PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.exchange.pairs.device
  SOURCES
  pairs_launch.cu
  kernels/record_bound/record_bound_00.cu
  kernels/record_bound/record_bound_01.cu
  kernels/record_bound/record_bound_02.cu
  kernels/record_bound/record_bound_10.cu
  kernels/record_bound/record_bound_11.cu
  kernels/record_bound/record_bound_12.cu
  kernels/record_bound/record_bound_20.cu
  kernels/record_bound/record_bound_21.cu
  kernels/record_bound/record_bound_22.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::exchange::pairs::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::basis::headers
  felizjueves::headers
  free_breakfast::integrals::sort::device
)

# The ordered classes, gathered on the host from mmd's, copied to the device
# and ordered there. MirroredBuffer, DeviceStream, DeviceResources, Result,
# BasisSet and PrimitivePairClass are in its interface, so their modules are
# PUBLIC; diagnostics.h (FBF_TRY) and the device library, whose bridge header
# the implementation unit includes in its GMF, are its only.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.exchange.pairs
  PRIMARY_INTERFACE
  pairs.cppm
  IMPLEMENTATION
  pairs.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::mirrored_buffer
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::integrals::exchange::pairs::headers
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::exchange::pairs::device
)
