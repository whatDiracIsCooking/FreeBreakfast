# The arithmetic header, pm_math.h, for the host tests: it is plain C++ outside
# a device pass, so test/gtest/pipek_mezey_host.cpp checks the localizer's
# algebra with no device. It reaches FBF_HOST_DEVICE and math:: through
# primitives.h.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.localization.pm.headers
  LINK_PUBLIC
  free_breakfast::core::primitives_header
)

# The kernels and their launchers. Device code in its own library, so its
# language and -x hip never reach the module compiles. The bridge header names
# int32_t, wwrStream_t and wwrError_t, hence types_headers and wwr::device
# PUBLIC; block_reduce and the launch shapes (kernel_headers) are its own
# business, as is pm_math.h.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.localization.pm.device
  SOURCES
  pm_kernels.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::localization::pm::headers
)

# The module. pipek_mezey_localize is declared in the interface and defined,
# with its two instantiations, in the implementation unit, which alone
# GMF-includes the bridge and pm_math.h. So the device library, the headers and
# what only the driver uses are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.localization.pm
  PRIMARY_INTERFACE
  pm.cppm
  IMPLEMENTATION
  pm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::localization::pm::headers
  free_breakfast::localization::pm::device
)
