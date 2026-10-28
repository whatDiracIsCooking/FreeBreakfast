# S² on a block of CI vectors, the diagonal preconditioner, the lowest-value
# select and the P-space's Hamiltonian: ci::solve's device work. Device code
# in its own library, so its language and -x hip never reach the module
# compiles. The bridge header it shares with the module names int32_t,
# Replacement, wwrStream_t and wwrError_t, hence types_headers,
# ci::strings::headers and wwr::device PUBLIC; block_params.h (kFourWarps),
# primitives.h (math::idivup), constants.h (kSqrt2, kInvSqrt2) from
# core::constants_header, and the sigma's ci_symmetric.h, reached through
# ci::sigma::device's include directory, are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.ci.solver.device
  SOURCES
  ci_solver.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::ci::strings::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::core::constants_header
  free_breakfast::ci::sigma::device
)

# ci::solve is a template over the caller's sigma, instantiated in the
# consumer's TU, so what it names -- davidson_solve, DeviceResources, the
# runtime's memcpy (core::wwr), DeviceStringSet, ci::Layout, ci::unpack and
# ci::pack (ci::sigma) -- is PUBLIC. The FBF_TRY macro
# expands in the interface unit, so its header is PRIVATE, and the
# implementation unit alone GMF-includes the bridge and calls orthonormalize,
# diagonalize (the P-space's) and the gemvs' kOne and kZero, so the device
# library, recipes::orthogonalize, recipes::diagonalize and core::constants
# are PRIVATE too.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.solver
  PRIMARY_INTERFACE
  solver.cppm
  IMPLEMENTATION
  solver.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::davidson
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::orthogonalize
  free_breakfast::ci::solver::device
)
