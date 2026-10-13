# Everything PUBLIC: this module is aliases over wwr's templates plus re-exports,
# so every name it hands out is instantiated in the consumer's TU.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.gpu
  PRIMARY_INTERFACE
  gpu.cppm
  LINK_PUBLIC
  free_breakfast::core::gpu_policy
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  wwr::extension::memory_buffer
  wwr::extension::blas
  wwr::extension::solver
  wwr::extension::tx
  IMPORT_STD
)
