# The S launcher. Device code in its own library, so its language and -x hip
# never reach the module compiles. Only overlap_launch.cu is a source: it
# #includes the generated kernels (kernels/), which must not also be compiled on
# their own (their extern "C" kernels would be defined twice). The bridge header
# it shares with the module names int32_t, kMaxAngularMomentum,
# PrimitivePairClassView and wwrStream_t, hence types_headers, basis::headers,
# upload::headers and wwr::device PUBLIC.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.overlap.device
  SOURCES
  overlap_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  free_breakfast::integrals::one_electron::upload::headers
  wwr::device
)

# The module. DeviceResources, DevicePrimitivePairClass and CartesianMatrixAO
# are in its signature, so core::resources, one_electron::upload and
# linalg::matrix_ao are PUBLIC. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.overlap
  PRIMARY_INTERFACE
  overlap.cppm
  IMPLEMENTATION
  overlap.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::one_electron::upload
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::overlap::device
)
