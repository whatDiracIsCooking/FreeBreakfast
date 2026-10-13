# diagnostics and wwr::extension::common are PUBLIC for the same reason
# diagnostics links common PUBLIC: the policies are templates over
# wwr::extension::error_type whose bodies (gpu_status, check_or_abort) are
# instantiated in the consumer's TU.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.gpu_policy
  PRIMARY_INTERFACE
  gpu_policy.cppm
  IMPLEMENTATION
  gpu_policy.cpp
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::diagnostics
  free_breakfast::core::log
  wwr::extension::common
  IMPORT_STD
)

# The ABORT/RECORD switch, resolved in the root CMakeLists. PRIVATE is enough --
# and is the point: the module reads it once into gpu::aborts_on_error, so
# consumers get the value through the BMI, never by seeing the macro themselves.
if(FREE_BREAKFAST_GPU_ERROR_POLICY_RESOLVED STREQUAL "ABORT")
  set(_fb_gpu_policy_aborts 1)
else()
  set(_fb_gpu_policy_aborts 0)
endif()
target_compile_definitions(
  free_breakfast.core.gpu_policy
  PRIVATE FREE_BREAKFAST_GPU_POLICY_ABORTS=${_fb_gpu_policy_aborts}
)
