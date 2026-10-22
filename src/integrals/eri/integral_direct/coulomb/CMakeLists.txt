# The ket sort the screen needs (its own device library), first: the module
# below links it.
add_subdirectory(sort_bounds)
# J's classes and the prune that makes them, before the three steps: each
# launches over them.
add_subdirectory(pair_class)
# The build's three generated steps, one kernel family each: the ket
# pre-contraction, the [p|q] contraction and the bra back-transform.
add_subdirectory(dq)
add_subdirectory(e_hermite)
add_subdirectory(j_coulomb)

# The module: the build, which runs the three steps and the ket sort ket-major.
# DeviceResources, CoulombPairClass (pair_class, re-exported) and
# CartesianMatrixAO are in its signature, so core::resources, pair_class and
# linalg::matrix_ao are PUBLIC. The per-class launch and the pre-launch checks
# are the one-electron builds' own (one_electron::launch), the three steps are
# modules of their own, and the ket sort's device library is PRIVATE: the
# implementation unit includes its bridge header in its GMF and calls the
# launcher, and a static library's private link dependencies still reach the
# final link. The momentum limit is basis's.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.coulomb
  PRIMARY_INTERFACE
  coulomb.cppm
  IMPLEMENTATION
  coulomb.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::integrals::coulomb::pair_class
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::coulomb::sort_bounds::device
  free_breakfast::integrals::coulomb::dq
  free_breakfast::integrals::coulomb::e_hermite
  free_breakfast::integrals::coulomb::j_coulomb
)
