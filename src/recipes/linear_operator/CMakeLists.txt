# The concept only: no implementation unit. Everything PUBLIC, since a
# consumer's TU checks the concept against DeviceResources and Status.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.linear_operator
  PRIMARY_INTERFACE
  linear_operator.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
)
