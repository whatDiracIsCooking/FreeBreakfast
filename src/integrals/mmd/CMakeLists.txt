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

# The record order: each class's records put in descending zeta order on the
# device, once per basis. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header names int32_t,
# wwrStream_t and wwrError_t, hence types_headers and wwr::device PUBLIC; the
# kernels read kFourWarps (block_params.h) and idivup (primitives.h), hence
# core::kernel_headers, and sort with src/integrals/sort's merge sort; both
# PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.mmd.device
  SOURCES
  mmd_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::integrals::sort::device
)

# The records, built on the host, copied to the device and ordered there.
# MirroredBuffer, DeviceStream, DeviceResources, Result and BasisSet are in its
# interface, so their modules are PUBLIC. The device library is PRIVATE: the
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
  free_breakfast::core::mirrored_buffer
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd::headers
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::mmd::device
)
