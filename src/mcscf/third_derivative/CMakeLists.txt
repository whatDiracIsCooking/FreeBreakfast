# The module: mcscf::ThirdDerivative, declared in the interface and defined in
# the implementation units, third_derivative.cpp (create, apply) and
# integral_derivative.cpp (apply_integrals, and df_integral_derivatives).
# DeviceResources, Buffers, the AO matrices, the strings, the sigma, the Rdm,
# Molecule, BasisSet and PrimitivePairClass are in its signatures or members,
# so core::wwr, core::resources, recipes::matrix_ao, ci::strings, ci::sigma,
# ci::rdm, basis, molecule and mmd are PUBLIC; the packed order's copies, the
# gemms, the derivative integrals and FBF_TRY are only used. It launches
# nothing of its own: sigmas, RDMs, BLAS, copies and the derivative calls, so
# there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.third_derivative
  PRIMARY_INTERFACE
  third_derivative.cppm
  IMPLEMENTATION
  third_derivative.cpp
  integral_derivative.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::matrix_ao
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::basis
  free_breakfast::molecule
  free_breakfast::integrals::mmd
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::response::orbital_pairs
  free_breakfast::recipes::blas
  free_breakfast::integrals::one_electron::gradient
  free_breakfast::integrals::eri::density_fitting::gradient
)
