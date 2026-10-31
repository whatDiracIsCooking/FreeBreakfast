# SC-NEVPT2 over integrals, ci and core, never pipeline: the density-fitted
# MO blocks and external-integral slabs, the reference state's RDMs, the class
# energies, and the correlation energy's adjoint over (h, B, c).
add_subdirectory(density_fitting)
add_subdirectory(rdm)
add_subdirectory(energy)
add_subdirectory(gradient)

# The aggregate: one import for the NEVPT2 layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2
  PRIMARY_INTERFACE
  nevpt2.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::rdm
  free_breakfast::nevpt2::energy
  free_breakfast::nevpt2::gradient
)
