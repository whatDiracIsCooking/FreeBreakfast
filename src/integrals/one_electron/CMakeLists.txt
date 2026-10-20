add_subdirectory(launch)
add_subdirectory(overlap)
add_subdirectory(kinetic)
add_subdirectory(potential_en)
# After kinetic and potential_en: H is built from both.
add_subdirectory(core_hamiltonian)

# The aggregate: one import for the one-electron layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron
  PRIMARY_INTERFACE
  one_electron.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::integrals::one_electron::launch
  free_breakfast::integrals::one_electron::overlap
  free_breakfast::integrals::one_electron::kinetic
  free_breakfast::integrals::one_electron::potential_en
  free_breakfast::integrals::one_electron::core_hamiltonian
)
