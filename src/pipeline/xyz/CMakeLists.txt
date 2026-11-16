# The Xyz step. atom and molecule are PUBLIC (re-exported: GeometryOutput holds
# Atoms, and OptimizeOutput, declared here for the step's load, a Molecule),
# and so are pipeline::step and pipeline::footprint, since View and
# MemoryFootprint are in its signatures. io::xyz is PUBLIC too: run's inline
# body calls read_xyz from the importer's TU. diagnostics_macros is only
# FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.xyz
  PRIMARY_INTERFACE
  xyz.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::atom
  free_breakfast::molecule
  free_breakfast::io::xyz
  free_breakfast::pipeline::step
  free_breakfast::pipeline::footprint
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
