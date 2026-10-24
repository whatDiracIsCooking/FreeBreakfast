# The PrimitivePairClasses step. integrals::mmd is PUBLIC (re-exported: the
# step's output is its PrimitivePairClass), and so are the basis_set step,
# pipeline::step and core::resources, since View<BasisSetOutput> and
# DeviceResources are in run's signature. diagnostics_macros is only FBF_TRY,
# expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.primitive_pair_classes
  PRIMARY_INTERFACE
  primitive_pair_classes.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::pipeline::basis_set
  free_breakfast::integrals::mmd
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
