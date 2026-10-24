# The BasisSet step. basis is PUBLIC (re-exported: the step's output holds a
# BasisSet), and so are the molecule step and pipeline::step, since
# View<MoleculeOutput> is in run's signature. io::basis is PUBLIC too: run's
# inline body calls read_basis from the importer's TU. diagnostics_macros is
# only FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.basis_set
  PRIMARY_INTERFACE
  basis_set.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::pipeline::molecule
  free_breakfast::basis
  free_breakfast::io::basis
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
