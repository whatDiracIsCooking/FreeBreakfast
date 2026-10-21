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

# The ordered classes, gathered on the host from mmd's and copied to the
# device: no device code of its own. MirroredBuffer, DeviceStream,
# DeviceResources, Result, BasisSet and PrimitivePairClass are in its
# interface, so their modules are PUBLIC; diagnostics.h (FBF_TRY) is the
# implementation unit's only.
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
)
