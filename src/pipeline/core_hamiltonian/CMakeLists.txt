# The CoreHamiltonian step. matrix_ao is PUBLIC (re-exported: the step's
# output is a SphericalMatrixAO), and so are the spherical_transform and
# primitive_pair_classes steps, molecule, basis, pipeline::step and
# core::resources, since their types are in run's signature.
# one_electron::core_hamiltonian is PUBLIC too: run's inline body calls
# core_hamiltonian from the importer's TU. diagnostics_macros is only FBF_TRY,
# expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.core_hamiltonian
  PRIMARY_INTERFACE
  core_hamiltonian.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::basis
  free_breakfast::linalg::matrix_ao
  free_breakfast::integrals::one_electron::core_hamiltonian
  free_breakfast::pipeline::step
  free_breakfast::pipeline::spherical_transform
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
