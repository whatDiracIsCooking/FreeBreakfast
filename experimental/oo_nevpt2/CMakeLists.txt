# The OO-NEVPT2 probe (design.html): quasi-Newton steps of E_CAS + E_PT2 in
# the orbitals, from a config's DF-CASSCF, over the NEVPT2 gradient's adjoint
# and Z-vector. A plain executable over the pipeline, not a ctest entry.
add_executable(free_breakfast_oo_nevpt2 main.cpp)
set_target_properties(
  free_breakfast_oo_nevpt2 PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_oo_nevpt2
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::core::diagnostics_macros
          free_breakfast::integrals
          free_breakfast::recipes
          free_breakfast::ci
          free_breakfast::response
          free_breakfast::mcscf
          free_breakfast::nevpt2
          free_breakfast::pipeline::config
)
