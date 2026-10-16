# The inverse_sqrt kernel. Device code in its own library, so its language and
# -x hip never reach the module compiles. The bridge header it shares with the
# module names int32_t and wwrStream_t, hence types_headers and wwr::device
# PUBLIC; parallel_for and primitives.h (math::sqrt) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.linalg.orthogonalize.device
  SOURCES
  inverse_sqrt.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::primitives_header
  wwr::extension::parallel_for
)

# The module. DeviceResources is in its signatures, so core::resources is
# PUBLIC. The device library is PRIVATE: the implementation unit includes the
# bridge header in its GMF and calls the launcher, and a static library's
# private link dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg.orthogonalize
  PRIMARY_INTERFACE
  orthogonalize.cppm
  IMPLEMENTATION
  orthogonalize.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::linalg::orthogonalize::device
)
