free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.third_party.taskflow
  PRIMARY_INTERFACE
  taskflow.cppm
  LINK_PUBLIC
  Taskflow::Taskflow
  IMPORT_STD)
