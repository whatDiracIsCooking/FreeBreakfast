# The module: nevpt2::sc_energy and nevpt2::sc_adjoint, z^c's
# zc_multipliers and zc_adjoint, and the nuclear gradient's
# lagrangian_gradient and sc_gradient; the internal partition :operands holds
# the array plumbing the implementation units share. ci::strings,
# nevpt2::energy (ClassEnergies), core::wwr (Buffers), and the molecule,
# bases, pair classes, SphericalMatrixAO, mcscf::Solution, ZVector and
# GradientTerms of mcscf::df_gradient's signature are in its signatures, so
# they are PUBLIC with core::resources and core::error_handling: zvector
# too, internal to the MCSCF layer but in ScGradient as it is in
# df_gradient's signature. The builders' einsums, the BLAS calls, b_mo,
# rdms, the density-fitted adjoints, ci::Sigma and ci::Rdm, the CASSCF's
# integrals, Fock and Hessian, and the MCSCF layer's internal
# relaxed_hessian are the implementation units' alone. No device code of
# its own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  PARTITIONS
  operands.cppm
  IMPLEMENTATION
  gradient.cpp
  zc.cpp
  lagrangian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::integrals::mmd
  free_breakfast::recipes::matrix_ao
  free_breakfast::ci::strings
  free_breakfast::mcscf::solve
  free_breakfast::mcscf::zvector
  free_breakfast::mcscf::gradient
  free_breakfast::nevpt2::energy
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::einsum
  free_breakfast::recipes::symmetrize
  free_breakfast::integrals::active_space
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::rdm
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::hessian
  free_breakfast::response::orbital_pairs
  free_breakfast::mcscf::relaxed_hessian
)
