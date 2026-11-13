# The routine spike (#1246): Routine<Self, Alloc>, AllocKey and KeyedBuffer
# (partitions :base and :key), and mcscf::root_rotation ported onto them
# (:root_rotation, its check, footprint, invoke and the root's run defined in
# root_rotation.cpp). Host code over the original's callees, with no kernels:
# the port links what free_breakfast.mcscf.root_rotation links, plus that
# module itself for the result type and kRootGapFloor.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.routine
  PRIMARY_INTERFACE
  routine.cppm
  PARTITIONS
  key.cppm
  base.cppm
  root_rotation.cppm
  IMPLEMENTATION
  root_rotation.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::core::footprint
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::recipes::matrix_ao
  free_breakfast::mcscf::derivative
  free_breakfast::mcscf::root_rotation
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::orbital_pairs
  free_breakfast::response::relaxed_hessian
)
