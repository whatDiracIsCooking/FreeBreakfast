# wwr::vector_types is PUBLIC because the module re-exports it: a point in space
# is the backend's own double3, which a consumer names as wwr::double3.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.molecule
  PRIMARY_INTERFACE
  molecule.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::atom
  wwr::vector_types
)
