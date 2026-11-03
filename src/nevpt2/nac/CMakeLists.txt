# The module: the QD-NEVPT2 PT2 transition density, SC and PC, over the
# effective Hamiltonians' amplitudes, and the QD-NEVPT2 derivative coupling
# d_PQ over it, the interstate coupling and mcscf::df_nac's combination.
# DeviceResources, MoBlocks, EnergyInputs, Buffers, Result and df_nac's
# Molecule, BasisSet, PrimitivePairClass, SphericalMatrixAO, Solution and
# NacOptions are in its signatures, so core::resources, core::wwr,
# nevpt2::density_fitting, nevpt2::energy, core::error_handling, molecule,
# basis, mmd, recipes::matrix_ao, mcscf::solve and mcscf::nac are PUBLIC.
# The einsum, nevpt2::gradient (the coupling and qd_inputs) and the FBF_TRY
# header are the implementation units' alone. No device code of its own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.nac
  PRIMARY_INTERFACE
  nac.cppm
  IMPLEMENTATION
  pt2_density.cpp
  coupling.cpp
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
  free_breakfast::mcscf::solve
  free_breakfast::mcscf::nac
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::energy
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::einsum
  free_breakfast::nevpt2::gradient
)
