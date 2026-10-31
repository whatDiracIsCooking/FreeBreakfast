# Density fitting: the Coulomb metric (P|Q) (metric/, a module over its
# generated kernels' device library), the three-center (mn|P) (three_center/,
# the same), the fit B = L^-1 (mn|P) (fit/, host code over the solver and
# BLAS), DF J and K over B (jk/), B in the MO basis (mo/, BLAS alone), and the
# nuclear derivatives of the metric and of (mn|P), contracted and not, with
# the DF J and K derivatives over them (gradient/, a module over its generated
# kernels' device library), and the contracted second derivatives of the
# metric and of (mn|P) (hessian/, the same).
add_subdirectory(fit)
add_subdirectory(gradient)
add_subdirectory(hessian)
add_subdirectory(jk)
add_subdirectory(metric)
add_subdirectory(mo)
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
  free_breakfast::integrals::eri::density_fitting::three_center
  free_breakfast::integrals::eri::density_fitting::fit
  free_breakfast::integrals::eri::density_fitting::jk
  free_breakfast::integrals::eri::density_fitting::mo
  free_breakfast::integrals::eri::density_fitting::gradient
  free_breakfast::integrals::eri::density_fitting::hessian
)
