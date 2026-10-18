# A class template defined in its interface, so every name it uses is
# instantiated in the consumer's TU: all links are PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.mirrored_buffer
  PRIMARY_INTERFACE
  mirrored_buffer.cppm
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::nontransferrable
  free_breakfast::core::wwr
  IMPORT_STD
)
