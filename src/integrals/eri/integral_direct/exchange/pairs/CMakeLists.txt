# OrderedPairClassView, the exchange launchers' view of an ordered class,
# offered two ways from one definition in ordered_pair_class_view.h: `#include`
# through this include path, and `import
# free_breakfast.integrals.eri.integral_direct.exchange.pairs;`, which includes
# it in its GMF and re-exports the name (mmd's headers library, the same
# arrangement).
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.exchange.pairs.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# The order, the gather and the Dcd rows: each ordered class's records grouped
# by first shell and sorted by mmd's dressed bound, then written straight from
# the mmd class, and its bra rows from them, once per basis. Device code in its
# own library, so its language and -x hip never reach the module compiles. The
# bridge header names int32_t,
# PrimitivePairClassView (mmd's headers), wwrStream_t and wwrError_t, hence
# types_headers, mmd::headers and wwr::device PUBLIC; the launcher reads
# kFourWarps and idivup (core::kernel_headers) and sorts with
# src/recipes/sort's merge sort; both PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.exchange.pairs.device
  SOURCES
  pairs_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::core::kernel_headers
  free_breakfast::recipes::sort::device
)

# The ordered classes: their groups and table sizes from the basis on the
# host, their records ordered and gathered on the device from mmd's, and the
# K build's block tables, made once per basis by the exchange helpers'
# launcher. StreamBuffers, DeviceStream, DeviceResources, Result, BasisSet and
# PrimitivePairClass are in its interface, so their modules are PUBLIC;
# diagnostics.h (FBF_TRY), the device library and the helpers, whose bridge
# headers the implementation unit includes in its GMF, are its only.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.exchange.pairs
  PRIMARY_INTERFACE
  pairs.cppm
  IMPLEMENTATION
  pairs.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::integrals::mmd
  free_breakfast::integrals::eri::integral_direct::exchange::pairs::headers
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::integral_direct::exchange::pairs::device
  free_breakfast::integrals::eri::integral_direct::exchange::helpers
)
