# The module: pack_orbital_pairs and unpack_orbital_pairs, declared in the
# interface and defined in the implementation unit. Status and DeviceResources
# are in their signatures, so core::error_handling and core::resources are
# PUBLIC. It launches nothing of its own: 2D copies, a memset and geams, so
# there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.response.orbital_pairs
  PRIMARY_INTERFACE
  orbital_pairs.cppm
  IMPLEMENTATION
  orbital_pairs.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::core::wwr
)
