# The SphericalTransform step. matrix_ao is PUBLIC (re-exported: the step's
# consumers build MatrixAOPairs over the buffer it uploads), and so are basis,
# pipeline::step and core::resources, since View<BasisSet> and DeviceResources
# are in run's signature, and core::wwr, for the Buffers in its output.
# diagnostics_macros is only FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.spherical_transform
  PRIMARY_INTERFACE
  spherical_transform.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
