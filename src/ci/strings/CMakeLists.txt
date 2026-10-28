# Replacement, one link of a replacement list, offered two ways from one
# definition in replacement.h: `#include` through this include path, for the
# device gathers, and `import free_breakfast.ci.strings;`, which includes it in
# its GMF and re-exports the name.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.ci.strings.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

# StringSet is host-only and over plain integers; DeviceStringSet uploads one,
# so the GPU consumers share its tables. core::error_handling is PUBLIC because
# create returns a Result, core::wwr and core::resources because
# DeviceStringSet holds Buffers and takes DeviceResources, and the headers
# library because the interface re-exports Replacement.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.strings
  PRIMARY_INTERFACE
  strings.cppm
  IMPLEMENTATION
  strings.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::ci::strings::headers
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
