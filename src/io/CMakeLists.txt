add_subdirectory(base)
add_subdirectory(json)
add_subdirectory(npy)
add_subdirectory(toml)
add_subdirectory(config)

# The aggregate: one import for the file-level I/O layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.io
  PRIMARY_INTERFACE
  io.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::io::base
  free_breakfast::io::config
  free_breakfast::io::json
  free_breakfast::io::npy
  free_breakfast::io::toml
)
