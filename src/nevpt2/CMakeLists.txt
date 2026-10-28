# SC-NEVPT2 over integrals, ci and core, never pipeline: the density-fitted
# MO blocks and external-integral slabs, and the reference state's RDMs.
add_subdirectory(density_fitting)
add_subdirectory(rdm)

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
)
