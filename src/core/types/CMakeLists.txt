# Host-side module wrapper. Its GMF includes vec.h, which reaches the backend's
# builtin vector types through WarpWraps' neutral src/vector_types.h, so it
# needs wwr::vector_types::host -- the WWR_GPU_BACKEND_* define, wwr's src/
# include root, and the vendor header's include path. PUBLIC, as wwr's own
# wwr.vector_types does: a consumer naming Vec<float, 3> names ::float3. The
# :integral partition re-exports the fixed-width integer types and size_t from
# integral.h. The :concepts partition re-exports wwr's real_fp / int_type rather
# than redefining them, so wwr::wrappers::common is PUBLIC too: the concepts are
# evaluated in the consumer's TU.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.types
  PRIMARY_INTERFACE
  types.cppm
  PARTITIONS
  integral.cppm
  concepts.cppm
  LINK_PUBLIC
  wwr::vector_types::host
  wwr::wrappers::common
  IMPORT_STD
)

# INTERFACE library exposing this directory's headers to device translation
# units, which #include them directly (a named module cannot be imported into a
# .cu): vec.h for the builtin vector types and integral.h for the fixed-width
# integer types.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.types_headers
  LINK_PUBLIC
  wwr::vector_types::host
)
