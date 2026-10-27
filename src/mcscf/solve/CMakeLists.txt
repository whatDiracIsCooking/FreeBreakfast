# The augmented Hessian's preconditioner kernel. Device code in its own
# library, so its language and -x hip never reach the module compiles. The
# bridge header it shares with the module names int32_t, wwrStream_t and
# wwrError_t, hence types_headers and wwr::device PUBLIC; block_params.h
# (kFourWarps) and primitives.h (math::idivup) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.mcscf.solve.device
  SOURCES
  ah_precondition.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
)

# The module: mcscf::solve, declared in the interface and defined in the
# implementation unit. Molecule, BasisSet, DeviceResources, Buffers, the two
# kinds of pair class, the AO matrices and ci::Roots (in the solution) are in
# its signature or result, so molecule, basis, core::wwr, core::resources,
# coulomb, exchange, recipes::matrix_ao and ci::solver are PUBLIC; the loop's
# pieces are PRIVATE. Its one kernel is the preconditioner's, through the
# device library, PRIVATE: the implementation unit alone GMF-includes the
# bridge. The integrals, CI, RDMs, Fock, Hessian and expm are their modules',
# the rest BLAS and copies.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.solve
  PRIMARY_INTERFACE
  solve.cppm
  IMPLEMENTATION
  solve.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::ci::solver
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::expm
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::rdm
  free_breakfast::response::fock
  free_breakfast::response::hessian
  free_breakfast::mcscf::solve::device
)
