# The chemical element enum (Element) and an atom (Atom<T>), offered two ways
# from a single definition in element.h / atom.h:
#
#   * CUDA / HIP / plain headers: `#include <atom.h>` (or <element.h>), reached
#     through the free_breakfast::atom::headers include path below.
#   * C++ modules: `import free_breakfast.atom;` -- atom.cppm includes the
#     headers in its global module fragment and re-exports the names, so
#     importers and #include-rs share one entity.
free_breakfast_add_interface_library(NAME free_breakfast.atom.headers)

free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.atom
  PRIMARY_INTERFACE
  atom.cppm
  LINK_PUBLIC
  free_breakfast::atom::headers
)
