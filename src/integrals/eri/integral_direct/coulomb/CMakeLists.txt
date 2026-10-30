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
# The nuclear gradient's own step, gjab; its raised [p|q] is e_hermite's.
add_subdirectory(gradient)

# The module: the build, which runs the three steps and the ket sort ket-major,
# and its nuclear gradient, which runs the same ket phase with the bra raised by
# one and then gjab. DeviceResources, CoulombPairClass (pair_class,
# re-exported), BasisSet, PrimitivePairClass, CartesianMatrixAO and
# SphericalMatrixAO are in its signatures, so core::resources, pair_class,
# basis, mmd and recipes::matrix_ao are PUBLIC. The per-class launch and the
# pre-launch checks are the one-electron builds' own (one_electron::launch), the
# three steps are modules of their own, and the device libraries are PRIVATE:
# the ket sort's, e_hermite's (the raised [p|q]) and gjab's, whose bridge
# headers the implementation unit includes in its GMF; a static library's
# private link dependencies still reach the final link. So is pair_class's, for
# J's momentum limit, kCoulombMaxMomentum (its bridge header).
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.coulomb
  PRIMARY_INTERFACE
  coulomb.cppm
  IMPLEMENTATION
  coulomb.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class
  free_breakfast::recipes::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::eri::integral_direct::coulomb::pair_class::device
  free_breakfast::integrals::eri::integral_direct::coulomb::sort_bounds::device
  free_breakfast::integrals::eri::integral_direct::coulomb::dq
  free_breakfast::integrals::eri::integral_direct::coulomb::e_hermite
  free_breakfast::integrals::eri::integral_direct::coulomb::e_hermite::device
  free_breakfast::integrals::eri::integral_direct::coulomb::j_coulomb
  free_breakfast::integrals::eri::integral_direct::coulomb::gradient::device
)
