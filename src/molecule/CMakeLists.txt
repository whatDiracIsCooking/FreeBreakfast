# free_breakfast::core::types is PUBLIC because the module re-exports it: a
# point in space is a Vec<double, 3>, which a consumer names without a second
# import.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.molecule
  PRIMARY_INTERFACE
  molecule.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::atom
)
