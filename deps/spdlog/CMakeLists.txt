free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.third_party.spdlog
  PRIMARY_INTERFACE
  spdlog.cppm
  LINK_PUBLIC
  spdlog::spdlog
  IMPORT_STD)
