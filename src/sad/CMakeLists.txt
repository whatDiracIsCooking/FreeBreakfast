# Pure host data: no GPU link, so the guess builds and tests on a device-less
# box. basis is PUBLIC because the interface takes a BasisSet and a span of Atom
# (basis re-exports molecule, which re-exports atom); error_handling because it
# returns Result; taskflow because sad_density's task form takes a FlowBuilder
# and returns a Task. Taskflow is host-only too, so the module stays GPU-free.
#
# The interface exports sad_density alone, in a serial and a task form. The
# aufbau fill and the atomic SCF under atomic_scf/ are internal partitions,
# imported only by sad.cpp; the README's Files table says which partition is
# which.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.sad
  PRIMARY_INTERFACE
  sad.cppm
  PARTITIONS
  aufbau.cppm
  atomic_scf/hermite.cppm
  atomic_scf/one_electron.cppm
  atomic_scf/eri.cppm
  atomic_scf/integrals.cppm
  atomic_scf/gemm.cppm
  atomic_scf/eigen.cppm
  atomic_scf/diis.cppm
  atomic_scf/angular_block.cppm
  atomic_scf/scf.cppm
  IMPLEMENTATION
  sad.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  free_breakfast::third_party::taskflow
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
)
