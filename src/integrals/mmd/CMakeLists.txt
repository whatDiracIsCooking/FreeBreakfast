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

# The records, built on the host and copied to the device. MirroredBuffer,
# DeviceStream, DeviceResources, Result and BasisSet are in its interface, so
# their modules are PUBLIC.
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
)
