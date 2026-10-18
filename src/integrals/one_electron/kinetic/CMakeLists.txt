# The T launcher. Device code in its own library, so its language and -x hip
# never reach the module compiles. Only kinetic_launch.cu is a source: it
# #includes the generated kernels (kernels/), which must not also be compiled on
# their own (their kernels would be defined twice). The bridge header it shares
# with the module names int32_t, kMaxAngularMomentum, PrimitivePairClassView and
# wwrStream_t, hence types_headers, basis::headers, upload::headers and
# wwr::device PUBLIC. The kernels include FelizJueves's felizjueves/runtime.h,
# which only the launcher sees, hence felizjueves::headers PRIVATE; so are its
# block sizing (block_params.h, primitives.h), hence core::kernel_headers, and
# its table (kernel_table.h), hence one_electron::kernel_table.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.kinetic.device
  SOURCES
  kinetic_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::basis::headers
  free_breakfast::integrals::one_electron::upload::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  felizjueves::headers
  free_breakfast::integrals::one_electron::kernel_table
)

# The module. DeviceResources, DevicePrimitivePairClass and CartesianMatrixAO
# are in its signature, so core::resources, one_electron::upload and
# linalg::matrix_ao are PUBLIC. The device library is PRIVATE: the
# implementation unit includes the bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron.kinetic
  PRIMARY_INTERFACE
  kinetic.cppm
  IMPLEMENTATION
  kinetic.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::one_electron::upload
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::kinetic::device
)
