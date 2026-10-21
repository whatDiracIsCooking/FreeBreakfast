# The merge sort, before mmd and coulomb: both sort with it.
add_subdirectory(sort)
add_subdirectory(mmd)
# Before the families whose kernels include its headers.
add_subdirectory(shared)
add_subdirectory(one_electron)
# After one_electron: J reuses its per-class launch and checks.
add_subdirectory(coulomb)
add_subdirectory(exchange)

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
  free_breakfast::integrals::coulomb
)
