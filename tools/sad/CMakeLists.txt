# free-breakfast-sad-stamps: the generator of data/sad/, run by hand (see
# main.cpp and data/sad/README.md). Built with the tree rather than behind an
# option: it is one TU, and a generator nothing builds rots unseen until the
# day the stamps need regenerating.
#
# It links sad but not io::sad -- the stamps it writes are io::sad's input, and
# it must only ever compute them. Not a module library, so it sets the
# module-scanning properties itself, as src/main.cpp does.
#
# Which sets --all regenerates is a module library of its own,
# free_breakfast::tools::sad, so that free_breakfast_io_tests_host can pin how
# it names each data/basis document (#1446).
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.tools.sad
  PRIMARY_INTERFACE
  sets.cppm
  IMPLEMENTATION
  sets.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  LINK_PRIVATE
  free_breakfast::io::json
  free_breakfast::core::diagnostics_macros
)

add_executable(free_breakfast_sad_stamps main.cpp)
set_target_properties(
  free_breakfast_sad_stamps
  PROPERTIES OUTPUT_NAME free-breakfast-sad-stamps CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_sad_stamps
  PRIVATE free_breakfast::core::types
          free_breakfast::core::error_handling
          free_breakfast::core::diagnostics_macros
          free_breakfast::basis
          free_breakfast::io::basis
          free_breakfast::io::npy
          free_breakfast::sad
          free_breakfast::tools::sad
)
