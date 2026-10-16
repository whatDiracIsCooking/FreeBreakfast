# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
# Add your own domain code as sibling directories here and wire them in below.
#
# core/ is the starting set, not a fixture: one module library that imports the
# WarpWraps layer and `std`, so a fresh clone has something that actually
# exercises the whole chain -- the three cmake/ macros, C++23 named modules,
# `import std`, and the GPU backend switch. Delete it once this project has real
# modules of its own, or grow it; nothing else depends on it staying small.
add_subdirectory(core)
add_subdirectory(atom)
add_subdirectory(molecule)
add_subdirectory(basis)
add_subdirectory(io)
add_subdirectory(sad)
add_subdirectory(linalg)
