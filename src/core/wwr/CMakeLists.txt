# Everything PUBLIC: this module is gpu_status, the policies and aliases over
# wwr's templates, plus re-exports, so every name it hands out is instantiated
# in the consumer's TU. This is the one list of wwr module targets the module
# tree links; core/types (below it) and the header/device code (wwr::device)
# are the exceptions, see README.md.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.wwr
  PRIMARY_INTERFACE
  wwr.cppm
  PARTITIONS
  status.cppm
  policy.cppm
  fork_stream.cppm
  device.cppm
  gemm.cppm
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::nontransferrable
  free_breakfast::core::error_handling
  free_breakfast::core::log
  wwr::runtime_api
  wwr::vector_types
  wwr::wrappers::common
  wwr::wrappers::blas
  wwr::wrappers::solver
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  wwr::extension::memory_buffer
  wwr::extension::blas
  wwr::extension::solver
  wwr::extension::tx
  IMPORT_STD
)
