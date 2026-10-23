add_subdirectory(base)
add_subdirectory(json)
add_subdirectory(npy)
add_subdirectory(toml)
add_subdirectory(basis)
add_subdirectory(molden)
add_subdirectory(sad)
add_subdirectory(xyz)

# The aggregate: one import for the file-level I/O layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.io
  PRIMARY_INTERFACE
  io.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::io::base
  free_breakfast::io::basis
  free_breakfast::io::json
  free_breakfast::io::molden
  free_breakfast::io::npy
  free_breakfast::io::sad
  free_breakfast::io::toml
  free_breakfast::io::xyz
)
