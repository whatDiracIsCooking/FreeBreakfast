# A C ABI over DeviceResources, device buffers and an fp64 gemm
# (ffi/fb_gemm.h), for the Rust crate beside it (Cargo.toml) to link.
#
# Static, not shared: /opt/spdlog's libspdlog.a is not built -fPIC (its
# thread-local is local-exec), so no shared object can carry it. A Rust
# binary links the static archives instead, and it needs CMake's whole link
# line for that -- the transitive archives, the vendor libraries, libc++ -- so
# a probe executable records it: its CXX_LINKER_LAUNCHER (record_link.sh) writes
# the line to fb_gemm.link and then links. build.rs replays it.

# A plain TU that imports modules, like the benches' executables, so the
# module-scanning and import-std properties are set by hand.
add_library(free_breakfast_rust_gemm_ffi STATIC ffi/fb_gemm.cpp)
add_library(
  free_breakfast::experimental::rust_gemm_ffi ALIAS free_breakfast_rust_gemm_ffi
)
set_target_properties(
  free_breakfast_rust_gemm_ffi PROPERTIES CXX_MODULE_STD 1
                                          CXX_SCAN_FOR_MODULES ON
)
target_include_directories(
  free_breakfast_rust_gemm_ffi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ffi
)
target_link_libraries(
  free_breakfast_rust_gemm_ffi
  PUBLIC free_breakfast::core::types free_breakfast::core::wwr
         free_breakfast::core::resources
)

# Links the archive exactly as a C++ consumer would, so the line it records is
# complete, and so a missing symbol fails here rather than in cargo.
add_executable(free_breakfast_rust_gemm_linkprobe ffi/linkprobe.cpp)
set(_fb_record_link
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/record_link.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/fb_gemm.link"
)
set_target_properties(
  free_breakfast_rust_gemm_linkprobe
  PROPERTIES CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
             CXX_LINKER_LAUNCHER "${_fb_record_link}"
)
target_link_libraries(
  free_breakfast_rust_gemm_linkprobe
  PRIVATE free_breakfast::experimental::rust_gemm_ffi
)
