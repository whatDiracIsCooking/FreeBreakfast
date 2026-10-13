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
  IMPLEMENTATION
  gpu_policy.cpp
  LINK_PUBLIC
  free_breakfast::core::log
  wwr::extension::common
  IMPORT_STD
)

# The ABORT/RECORD switch, resolved in the root CMakeLists. PRIVATE is enough --
# and is the point: :gpu_policy reads it once into
# free_breakfast::kAbortsOnError, so consumers get the value through the BMI,
# never by seeing the macro themselves.
if(FREE_BREAKFAST_GPU_ERROR_POLICY_RESOLVED STREQUAL "ABORT")
  set(_fb_gpu_policy_aborts 1)
else()
  set(_fb_gpu_policy_aborts 0)
endif()
target_compile_definitions(
  free_breakfast.core.error_handling
  PRIVATE FREE_BREAKFAST_GPU_POLICY_ABORTS=${_fb_gpu_policy_aborts}
)
