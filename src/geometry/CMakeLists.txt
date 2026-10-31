# The Cartesian minimizer, host-only over a caller's energy and gradient, and
# the harmonic analysis of a Cartesian Hessian, whose two eigensolves are
# recipes' on the GPU. Result, the error kinds, DeviceResources and Atom are in
# the signatures, so their modules are PUBLIC; FBF_TRY, the BLAS and the
# eigensolver are the implementation units' own business. :rigid_body is an
# internal partition: the rigid-body basis both units project with.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.geometry
  PRIMARY_INTERFACE
  geometry.cppm
  PARTITIONS
  rigid_body.cppm
  IMPLEMENTATION
  geometry.cpp
  frequencies.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::atom
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::core::diagnostics_macros
)
