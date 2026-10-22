# The prune's launchers: the bound maxima, the keep flag and its scan, the
# scatter. Device code in its own library, so its language and -x hip never
# reach the module compiles. Nothing here launches another TU's kernel, so the
# library needs no relocatable device code, but keeps the default as the
# steps' do. The bridge header names int32_t, PrimitivePairClassView,
# wwrStream_t and wwrError_t, hence types_headers, mmd::headers and
# wwr::device PUBLIC. The .cu reads basis.h's function_count_cart (K), and
# block_reduce, kWarpSize and idivup (block_reduce.cuh, block_params.h,
# primitives.h), hence basis::headers and core::kernel_headers PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.coulomb.pair_class.device
  SOURCES
  pair_class_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  free_breakfast::integrals::mmd::headers
  wwr::device
  LINK_PRIVATE
  free_breakfast::basis::headers
  free_breakfast::core::kernel_headers
)

# The module. Its class holds StreamBuffers on a DeviceStream and an int32_t,
# and coulomb_pair_classes takes DeviceResources and PrimitivePairClass and
# returns a Result, so core::types, core::wwr, core::resources,
# core::error_handling and mmd are PUBLIC. The per-class launch is the
# one-electron builds' (one_electron::launch), and the device library is
# PRIVATE: the implementation unit includes its bridge header in its GMF and
# calls the launchers. The momentum limit is basis's.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.coulomb.pair_class
  PRIMARY_INTERFACE
  pair_class.cppm
  IMPLEMENTATION
  pair_class.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::integrals::mmd
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::basis
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::coulomb::pair_class::device
)
