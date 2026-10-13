# log and wwr::extension::common are PUBLIC because :gpu_status and :gpu_policy
# are templates over wwr::extension::error_type: their bodies (success_code /
# error_name / error_string, log::critical / log::error) are instantiated in
# the consumer's TU, so the consumer must be able to reach those names too.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.error_handling
  PRIMARY_INTERFACE
  error_handling.cppm
  PARTITIONS
  error.cppm
  gpu_status.cppm
  gpu_policy.cppm
  LINK_PUBLIC
  free_breakfast::core::log
  wwr::extension::common
  IMPORT_STD
)
