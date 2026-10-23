# The PrimitivePairClasses step. integrals::mmd is PUBLIC (re-exported: the
# step's output is its PrimitivePairClass), and so are basis, pipeline::step
# and core::resources, since View<BasisSet> and DeviceResources are in run's
# signature. core::wwr is gpu_status, for the wait on each class's stream;
# diagnostics_macros is only FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.primitive_pair_classes
  PRIMARY_INTERFACE
  primitive_pair_classes.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
