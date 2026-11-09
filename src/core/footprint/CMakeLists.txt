# Templates and constexpr functions only, so no implementation unit: every
# name they use is instantiated in the consumer's TU, hence all PUBLIC.
# core.wwr is linked for wwr::extension::MemoryKind, the footprint's kind.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.footprint
  PRIMARY_INTERFACE
  footprint.cppm
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::wwr
  IMPORT_STD
)
