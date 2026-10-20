# PairFields, the record's fields named once, offered two ways from one
# definition in pair_fields.h: `#include <pair_fields.h>` (through
# one_electron.upload's device_views.h, for the launchers) through this include
# path, and `import free_breakfast.integrals.mmd;`, which includes it in its
# GMF and re-exports the name.
free_breakfast_add_interface_library(NAME free_breakfast.integrals.mmd.headers)

# Pure host data: no GPU link, so the records build and are tested on a
# device-less box. basis is PUBLIC because the interface takes a BasisSet.
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
  free_breakfast::basis
  free_breakfast::integrals::mmd::headers
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::error_handling
)
