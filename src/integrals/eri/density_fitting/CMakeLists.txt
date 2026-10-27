# Density fitting: the Coulomb metric (P|Q) (metric/, a module over its
# generated kernels' device library), the fit B = L^-1 (mn|P) (fit/, host code
# over the solver and BLAS), DF J and K over B (jk/), and the three-center
# (mn|P)'s generated kernels (three_center/, a device library only, as yet).
add_subdirectory(fit)
add_subdirectory(jk)
add_subdirectory(metric)
add_subdirectory(three_center)

# The aggregate: one import for density fitting.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting
  PRIMARY_INTERFACE
  density_fitting.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::integrals::eri::density_fitting::metric
  free_breakfast::integrals::eri::density_fitting::fit
  free_breakfast::integrals::eri::density_fitting::jk
)
