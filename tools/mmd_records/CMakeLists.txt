# free-breakfast-mmd-records: the generator of rust/integrals/mmd/tests/data/,
# the C++ tree's primitive-pair records for the Rust port's tier-4 test, run
# by hand on a GPU (see main.cpp and that directory's README.md). Built with
# the tree, as tools/sad is: a generator nothing builds rots unseen until the
# day its data needs regenerating.
#
# Not a module library, so it sets the module-scanning properties itself, as
# src/main.cpp does.
add_executable(free_breakfast_mmd_records main.cpp)
set_target_properties(
  free_breakfast_mmd_records
  PROPERTIES OUTPUT_NAME free-breakfast-mmd-records CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_mmd_records
  PRIVATE free_breakfast::core::types
          free_breakfast::core::error_handling
          free_breakfast::core::wwr
          free_breakfast::core::resources
          free_breakfast::core::diagnostics_macros
          free_breakfast::atom
          free_breakfast::molecule
          free_breakfast::basis
          free_breakfast::io::basis
          free_breakfast::integrals::mmd
)
