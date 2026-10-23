# The line searches' single-block kernels: Table 1's interpolation, Table 2's
# windowed DFT and step choice (line_search.cu), and the selection of the
# crossings among polynomial_roots' roots (select_root.cu). Device code in its
# own library, so its language and -x hip never reach the module compiles. The
# bridge header it shares with the module names int32_t, wwrStream_t and
# wwrError_t, hence types_headers and wwr::device PUBLIC; primitives_header
# (free_breakfast::math) and constants_header (kPi, kTwo) are its own business.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.recipes.cg_unitary.device
  SOURCES
  line_search.cu
  select_root.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::primitives_header
  free_breakfast::core::constants_header
)

# The module. cg_unitary is a template over the caller's cost function,
# instantiated in the consumer's TU, so everything its body names is PUBLIC:
# wwr's BLAS and ScopedPointerMode (core::wwr), kOne / kNegOne / kPi
# (core::constants), the log sink, skew_eig / skew_expm (recipes::expm). The
# FBF_TRY / FBF_LOG_* macros expand in this interface unit, so their header is
# PRIVATE. So is the device library: the implementation unit GMF-includes its
# bridge header and calls the launchers, as it alone calls polynomial_roots,
# and the interface GMF-includes cg_unitary_limits.h from its directory.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.recipes.cg_unitary
  PRIMARY_INTERFACE
  cg_unitary.cppm
  IMPLEMENTATION
  cg_unitary.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::constants
  free_breakfast::core::error_handling
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::expm
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::polynomial_roots
  free_breakfast::recipes::cg_unitary::device
)
