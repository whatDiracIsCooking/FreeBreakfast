add_subdirectory(mmd)
# Before the families whose kernels include its headers.
add_subdirectory(shared)
add_subdirectory(one_electron)
# After one_electron: J reuses its per-class launch and checks.
add_subdirectory(eri/integral_direct/coulomb)
add_subdirectory(eri/integral_direct/exchange)
# After J and K: a non-symmetric density's J - K/2 and K, built from both.
add_subdirectory(eri/integral_direct/nonsymmetric_jk)
# Density fitting, after shared: its generated kernels include its
# eval_boys.h.
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
  free_breakfast::integrals::eri::integral_direct::nonsymmetric_jk
  free_breakfast::integrals::eri::density_fitting
  free_breakfast::integrals::active_space
)
