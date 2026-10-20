# NucleiView, offered two ways from a single definition in nuclei_view.h:
# `#include <nuclei_view.h>` for the launchers, through this include path, and
# `import free_breakfast.molecule;`, which includes it in its GMF and
# re-exports the name. nuclei_view.h names int32_t through integral_types.h.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.molecule.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# free_breakfast::core::types is PUBLIC because the module re-exports it: a
# point in space is a Vec<double, 3>, which a consumer names without a second
# import. A Molecule owns its nuclei on the device, so DeviceResources,
# MirroredBuffer and DeviceStream are in its signatures and inline bodies:
# core::resources, core::mirrored_buffer and core::wwr are PUBLIC, and so is
# core::error_handling, for the constructor's check_or_abort.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.molecule
  PRIMARY_INTERFACE
  molecule.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::mirrored_buffer
  free_breakfast::core::resources
  free_breakfast::atom
  free_breakfast::molecule::headers
)
