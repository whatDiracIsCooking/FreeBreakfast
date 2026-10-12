free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.third_party.nlohmann
  PRIMARY_INTERFACE
  nlohmann.cppm
  LINK_PUBLIC
  nlohmann_json::nlohmann_json
  IMPORT_STD)
