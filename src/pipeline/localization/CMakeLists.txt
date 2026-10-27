# The Localization step. localization::ibo (with the pipek_mezey it re-exports),
# analysis::population and analysis::bond, core::wwr and pipeline::scf are
# PUBLIC and re-exported: the product holds Buffers and BondOrbitals, the
# options are localization::PmOptions, and run's View names the ScfOutput and
# the products it re-exports. core::resources, the basis_set step and
# pipeline::step are in run's signature. run is defined in the implementation
# unit, so
# io::basis (MINAO), localization::livvo, the Molden writer, recipes::blas and
# recipes::sort (the energy order of the localized orbitals) and
# diagnostics_macros (FBF_TRY) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.localization
  PRIMARY_INTERFACE
  localization.cppm
  IMPLEMENTATION
  localization.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::pipeline::basis_set
  free_breakfast::localization::pipek_mezey
  free_breakfast::localization::ibo
  free_breakfast::analysis::population
  free_breakfast::analysis::bond
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::atom
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::basis
  free_breakfast::io::molden
  free_breakfast::localization::livvo
  free_breakfast::recipes::blas
  free_breakfast::recipes::sort
)
