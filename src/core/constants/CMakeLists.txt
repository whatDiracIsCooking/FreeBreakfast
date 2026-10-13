free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.constants
  PRIMARY_INTERFACE
  constants.cppm
)

# constants.h is plain C++ with no dependencies, so a device translation unit
# (which cannot import a named module) #includes it directly through this
# target.
free_breakfast_add_interface_library(NAME free_breakfast.core.constants_header)
