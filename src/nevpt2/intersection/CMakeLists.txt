# The module: the QD-NEVPT2 branching space of one QD pair, SC and PC, and
# its gap sensitivities, over the gradient layer's W forms. DeviceResources,
# Buffers, Result, MemoryFootprint and the W forms' Molecule, BasisSet,
# PrimitivePairClass, SphericalMatrixAO and Solution are in its signatures, so
# core::resources, core::wwr, core::error_handling, core::footprint, molecule,
# basis, mmd, recipes::matrix_ao and mcscf::solve are PUBLIC.
# nevpt2::gradient (qd_inputs and the W forms), nevpt2::energy (the effective
# Hamiltonians), H_eff's diagonalize, mcscf::root_rotation (kRootGapFloor) and
# the FBF_TRY header are the implementation unit's alone. No device code of
# its own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2.intersection
  PRIMARY_INTERFACE
  intersection.cppm
  IMPLEMENTATION
  intersection.cpp
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
  free_breakfast::core::footprint
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::diagonalize
  free_breakfast::mcscf::root_rotation
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::energy
  free_breakfast::nevpt2::gradient
)
