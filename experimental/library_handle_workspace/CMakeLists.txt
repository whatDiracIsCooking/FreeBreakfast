# Does a vendor solver or BLAS handle allocate device memory behind a
# caller-sized workspace? One case per process; run_all.sh runs them. A plain
# executable over wwr's raw solver and BLAS names (on HIP, hipSOLVER and
# hipBLAS over rocSOLVER and rocBLAS), not a ctest entry.
add_executable(
  free_breakfast_library_handle_workspace library_handle_workspace.cpp
)
set_target_properties(
  free_breakfast_library_handle_workspace
  PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_library_handle_workspace PRIVATE wwr::runtime_api wwr::blas
                                                  wwr::solver
)
