# PrimitivePairClassView, the launchers' view of a class, offered two ways from
# one definition in primitive_pair_class_view.h: `#include` (through the
# launchers' bridge headers) through this include
# path, and `import free_breakfast.integrals.mmd;`, which includes it in its
# GMF and re-exports the name.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.mmd.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# The records, their order and their bounds: each class's records formed on
# the device from the basis's shells and the class's shell pairs, in
# descending zeta order, then each record's Schwarz bound and its dressed
# bound, as stored and swapped, once per basis. Device code in its own
# library, so its language and -x hip never reach the module compiles. Its
# sources are the
# launcher and the generated kernels, one per class, each compiled on its own:
# pair_bound (kernels/pair_bound/, written by
# tools/emit_one_electron_kernels.py) and record_bound (kernels/record_bound/,
# written by tools/emit_exchange_kernels.py). Under CUDA that wants
# relocatable device code, this library's default, so it must never pass
# NO_DEVICE_LINKING. Only mmd's own module calls the launchers. The bridge
# header names int32_t, PrimitivePairClassView, wwrStream_t and
# wwrError_t, hence types_headers, headers and wwr::device PUBLIC; the kernels
# read kFourWarps (block_params.h) and idivup and pow (primitives.h), hence
# core::kernel_headers, kPi (constants.h), hence core::constants_header,
# sort with src/recipes/sort's merge sort, size the
# bounds' tables by kMaxAngularMomentum (basis::headers), and the generated
# units include felizjueves/runtime.h (felizjueves::headers); all PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.mmd.device
  SOURCES
  mmd_launch.cu
  kernels/pair_bound/pair_bound_00.cu
  kernels/pair_bound/pair_bound_01.cu
  kernels/pair_bound/pair_bound_02.cu
  kernels/pair_bound/pair_bound_11.cu
  kernels/pair_bound/pair_bound_12.cu
  kernels/pair_bound/pair_bound_22.cu
  kernels/record_bound/record_bound_00.cu
  kernels/record_bound/record_bound_01.cu
  kernels/record_bound/record_bound_02.cu
  kernels/record_bound/record_bound_03.cu
  kernels/record_bound/record_bound_10.cu
  kernels/record_bound/record_bound_11.cu
  kernels/record_bound/record_bound_12.cu
  kernels/record_bound/record_bound_13.cu
  kernels/record_bound/record_bound_20.cu
  kernels/record_bound/record_bound_21.cu
  kernels/record_bound/record_bound_22.cu
  kernels/record_bound/record_bound_23.cu
  kernels/record_bound/record_bound_30.cu
  kernels/record_bound/record_bound_31.cu
  kernels/record_bound/record_bound_32.cu
  kernels/record_bound/record_bound_33.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
  free_breakfast::basis::headers
  felizjueves::headers
  free_breakfast::recipes::sort::device
)

# The classes: the shell-pair sweep on the host, the shell and shell-pair
# tables staged there and uploaded, everything else on the device.
# StreamBuffers, DeviceStream, DeviceResources, Result and BasisSet are in its
# interface, so their modules are PUBLIC; MirroredBuffer stages the tables in
# the implementation unit only. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launchers.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.mmd
  PRIMARY_INTERFACE
  mmd.cppm
  IMPLEMENTATION
  mmd.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd::headers
  LINK_PRIVATE
  free_breakfast::core::mirrored_buffer
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::mmd::device
)
