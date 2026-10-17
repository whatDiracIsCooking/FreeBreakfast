# The S, T and V launchers. Device code in its own library, so its language and
# -x hip never reach the module compiles. Only the three *_launch.cu are
# sources: each #includes its family's generated kernels (s_overlap/,
# t_kinetic/, v_nuclear/), which must not also be compiled on their own (their
# extern "C" kernels would be defined twice). The bridge header they share with
# the module names int32_t and wwrStream_t, hence types_headers and wwr::device
# PUBLIC.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.one_electron.device
  SOURCES
  overlap_launch.cu
  kinetic_launch.cu
  nuclear_launch.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  wwr::device
)

# The module. DeviceResources, Buffers, PrimitivePairClass, Molecule and
# CartesianMatrixAO are in its signatures, so core::resources, core::wwr,
# integrals::mmd, molecule and linalg::matrix_ao are PUBLIC. The device library
# is PRIVATE: the implementation unit includes the bridge header in its GMF and
# calls the launchers, and a static library's private link dependencies still
# reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron
  PRIMARY_INTERFACE
  one_electron.cppm
  IMPLEMENTATION
  one_electron.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::integrals::mmd
  free_breakfast::molecule
  free_breakfast::linalg::matrix_ao
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::one_electron::device
)
