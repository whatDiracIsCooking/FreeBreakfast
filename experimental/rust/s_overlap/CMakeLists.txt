# A C ABI over the pipeline's setup steps and its overlap step
# (ffi/fb_overlap.h): S of a molecule in a basis, for the Rust crate beside it
# (Cargo.toml).
#
# Static, not shared: /opt/spdlog's libspdlog.a is not built -fPIC (its
# thread-local is local-exec), so no shared object can carry it. A Rust
# binary links the static archives instead, and it needs CMake's whole link
# line for that -- the transitive archives, the vendor libraries, libc++ -- so
# a probe executable records it: its CXX_LINKER_LAUNCHER (ffi/record_link.sh)
# writes the line to fb_overlap.link and then links. build.rs replays it.

# A plain TU that imports modules, like the benches' executables, so the
# module-scanning and import-std properties are set by hand.
add_library(free_breakfast_rust_s_overlap_ffi STATIC ffi/fb_overlap.cpp)
add_library(
  free_breakfast::experimental::rust_s_overlap_ffi ALIAS
  free_breakfast_rust_s_overlap_ffi
)
set_target_properties(
  free_breakfast_rust_s_overlap_ffi PROPERTIES CXX_MODULE_STD 1
                                               CXX_SCAN_FOR_MODULES ON
)
target_include_directories(
  free_breakfast_rust_s_overlap_ffi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ffi
)
target_link_libraries(
  free_breakfast_rust_s_overlap_ffi
  PUBLIC free_breakfast::core::error_handling
         free_breakfast::core::diagnostics_macros
         free_breakfast::core::wwr
         free_breakfast::core::resources
         free_breakfast::pipeline::step
         free_breakfast::pipeline::molecule
         free_breakfast::pipeline::basis_set
         free_breakfast::pipeline::spherical_transform
         free_breakfast::pipeline::primitive_pair_classes
         free_breakfast::pipeline::overlap
)

# Links the archive exactly as a C++ consumer would, so the line it records is
# complete, and so a missing symbol fails here rather than in cargo.
add_executable(free_breakfast_rust_s_overlap_linkprobe ffi/linkprobe.cpp)
set(_fb_record_overlap_link
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/record_link.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/fb_overlap.link"
)
set_target_properties(
  free_breakfast_rust_s_overlap_linkprobe
  PROPERTIES CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
             CXX_LINKER_LAUNCHER "${_fb_record_overlap_link}"
)
target_link_libraries(
  free_breakfast_rust_s_overlap_linkprobe
  PRIVATE free_breakfast::experimental::rust_s_overlap_ffi
)
# The crate depends on rust/gemm's, which links WarpWraps' C ABI: one target
# builds everything cargo needs.
add_dependencies(
  free_breakfast_rust_s_overlap_linkprobe wwr.capi.runtime_api wwr.capi.blas
)
