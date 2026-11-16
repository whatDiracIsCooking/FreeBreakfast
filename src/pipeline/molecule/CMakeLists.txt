# The Molecule step. molecule and pipeline::xyz are PUBLIC (re-exported: the
# step's output holds a Molecule, and it reads the Xyz step's GeometryOutput),
# and so are pipeline::step and core::resources, since View and
# DeviceResources are in run's signature.
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
  free_breakfast::pipeline::xyz
  free_breakfast::pipeline::step
  free_breakfast::pipeline::footprint
)
