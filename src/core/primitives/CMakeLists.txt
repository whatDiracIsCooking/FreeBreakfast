# Host-side module: GMF-includes primitives.h and re-exports its names. The
# header names int32_t through integral_types.h, hence types_headers; it uses
# <math.h> and <type_traits> rather than `import std`, so no IMPORT_STD.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.primitives
  PRIMARY_INTERFACE
  primitives.cppm
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# INTERFACE library exposing primitives.h to device translation units, which
# #include it directly (a named module cannot be imported into a .cu). In a
# device pass the header includes wwr's runtime.h, which selects the vendor
# runtime header; wwr::device carries both include paths (CUDA::cudart or
# hip::host, never hip::device).
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.primitives_header
  LINK_PUBLIC
  wwr::device
  free_breakfast::core::types_headers
)
