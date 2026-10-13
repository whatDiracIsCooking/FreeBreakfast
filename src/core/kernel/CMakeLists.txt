# Host-side module: GMF-includes block_params.h and re-exports its names. The
# header reads WWR_WARP_SIZE, which wwr::device puts on the command line, and
# names uint32_t through integral_types.h, hence types_headers. PRIVATE
# wwr::device: importers see the finished kWarpSize, never the macro. No
# IMPORT_STD; the header uses <concepts>.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.kernel
  PRIMARY_INTERFACE
  kernel.cppm
  LINK_PUBLIC
  free_breakfast::core::types_headers
  LINK_PRIVATE
  wwr::device
)

# INTERFACE library exposing this directory's headers to device translation
# units, which #include them directly (a named module cannot be imported into a
# .cu): block_params.h, device_functor.h, warp_reduce.cuh and block_reduce.cuh.
# wwr::device brings WWR_WARP_SIZE and the include path of device_guard.h,
# cooperative_groups.h and runtime.h; block_reduce.cuh reaches math::idivup
# through primitives.h.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.kernel_headers
  LINK_PUBLIC
  wwr::device
  free_breakfast::core::primitives_header
  free_breakfast::core::types_headers
)
