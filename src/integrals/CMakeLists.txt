add_subdirectory(mmd)
# Before the families whose kernels include its headers.
add_subdirectory(shared)
add_subdirectory(one_electron)
# After one_electron: J reuses its per-class launch and checks.
add_subdirectory(eri/integral_direct/coulomb)
add_subdirectory(eri/integral_direct/exchange)
# Density fitting's generated kernels (device libraries only, as yet), after
# shared: they include its eval_boys.h.
add_subdirectory(eri/density_fitting)
# After J and K, which it is built from.
add_subdirectory(active_space)

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
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::integrals::active_space
)
