# The Molecule step. molecule is PUBLIC (re-exported: it is the step's
# output), and so are pipeline::step and core::resources, since View and
# DeviceResources are in run's signature. io::xyz is PUBLIC too: run's inline
# body calls read_xyz from the importer's TU.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.molecule
  PRIMARY_INTERFACE
  molecule.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::io::xyz
  free_breakfast::pipeline::step
)
