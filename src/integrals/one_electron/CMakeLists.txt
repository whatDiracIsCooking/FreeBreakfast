add_subdirectory(upload)
add_subdirectory(overlap)
# kinetic, potential_en and core_hamiltonian are disabled: their sources stay in
# the tree but are not built, linked or re-exported (one_electron.cppm), and
# their test (core_hamiltonian_gpu.cpp) is off. To restore, uncomment these, the
# three links below, the three imports in one_electron.cppm and the test.
# add_subdirectory(kinetic)
# add_subdirectory(potential_en)
# After kinetic and potential_en: H is built from both.
# add_subdirectory(core_hamiltonian)

# The aggregate: one import for the one-electron layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.one_electron
  PRIMARY_INTERFACE
  one_electron.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::integrals::one_electron::upload
  free_breakfast::integrals::one_electron::overlap
  # free_breakfast::integrals::one_electron::kinetic
  # free_breakfast::integrals::one_electron::potential_en
  # free_breakfast::integrals::one_electron::core_hamiltonian
)
