# Re-exports wwr's real_fp / int_type rather than redefining them, so PUBLIC:
# the concepts are evaluated in the consumer's TU.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.concepts
  PRIMARY_INTERFACE
  concepts.cppm
  LINK_PUBLIC
  wwr::wrappers::common
  IMPORT_STD
)
