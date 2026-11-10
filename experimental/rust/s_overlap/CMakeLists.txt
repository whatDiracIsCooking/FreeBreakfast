# A C ABI over the pipeline's setup steps and its overlap step
# (ffi/fb_overlap.h): S of a molecule in a basis, for the Rust crate beside it
# (Cargo.toml). Built on rust/gemm's handles and linked like it: a static
# archive, and a probe whose link line rust/gemm's record_link.sh records for
# build.rs to replay. rust/gemm/CMakeLists.txt has why it is not a shared
# library.

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
  PUBLIC free_breakfast::experimental::rust_gemm_ffi
         free_breakfast::core::error_handling
         free_breakfast::core::diagnostics_macros
         free_breakfast::pipeline::step
         free_breakfast::pipeline::molecule
         free_breakfast::pipeline::basis_set
         free_breakfast::pipeline::spherical_transform
         free_breakfast::pipeline::primitive_pair_classes
         free_breakfast::pipeline::overlap
)

add_executable(free_breakfast_rust_s_overlap_linkprobe ffi/linkprobe.cpp)
set(_fb_record_overlap_link
    "${CMAKE_CURRENT_SOURCE_DIR}/../gemm/ffi/record_link.sh"
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
# The crate depends on rust/gemm's, whose build.rs reads gemm's own record:
# one target builds both.
add_dependencies(
  free_breakfast_rust_s_overlap_linkprobe free_breakfast_rust_gemm_linkprobe
)
