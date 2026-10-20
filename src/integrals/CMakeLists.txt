add_subdirectory(mmd)
add_subdirectory(one_electron)
# Device code only so far: the J-build kernels, compiled, launched by nothing.
add_subdirectory(coulomb)

# The aggregate: one import for the integral layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals
  PRIMARY_INTERFACE
  integrals.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::integrals::mmd
  free_breakfast::integrals::one_electron
)
