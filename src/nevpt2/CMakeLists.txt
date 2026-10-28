# SC-NEVPT2 over integrals and core, never pipeline: the density-fitted MO
# blocks and the external-integral slabs the class energies read.
add_subdirectory(density_fitting)

# The aggregate: one import for the NEVPT2 layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.nevpt2
  PRIMARY_INTERFACE
  nevpt2.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::nevpt2::density_fitting
)
