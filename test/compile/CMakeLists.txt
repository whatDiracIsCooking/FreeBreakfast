# ---------------------------------------------------------------------------
# The compile-time tier: TUs that are nothing but compile-time claims
# ---------------------------------------------------------------------------
# docs/testing.md "Which tier a check goes in": a claim the compiler can settle
# (a concept, a type, a layout, constexpr integer arithmetic, a refusal of the
# wrong type) is a static_assert, and a TU that holds only such claims is a
# <tree>_compile.cpp here. It declares no gtest case and links no GoogleTest,
# so it fails the build rather than a run, and ctest never lists it.
#
# The root CMakeLists adds this directory outside the
# FREE_BREAKFAST_COMPILE_TIME_ONLY gate that skips test/, beside
# test/core/gpu_kernels, so the compile-only configure builds it too, and
# devtools/cross-backend-check.sh checks it for the other backend.
#
# One OBJECT library for the tree rather than one per test tree: the file's
# prefix names its subject. A static_assert that guards a runtime case's own
# setup (a table of what a call takes, which the case then loops over) stays in
# that case's *_host.cpp or *_gpu.cpp; only a TU with no case comes here.
#
#   * recipes_compile.cpp -- the concepts src/recipes exports.
#   * footprint_compile.cpp -- the footprint kinds (src/core/footprint) and
#     the footprint member's detection (src/pipeline/footprint).
add_library(
  free_breakfast_compile_checks OBJECT recipes_compile.cpp
                                       footprint_compile.cpp
)

set_target_properties(
  free_breakfast_compile_checks PROPERTIES CXX_MODULE_STD 1
                                           CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_compile_checks
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::recipes
          free_breakfast::pipeline::footprint
)
