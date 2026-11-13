# ---------------------------------------------------------------------------
# Tests of experimental/
# ---------------------------------------------------------------------------
# Added by the root CMakeLists only under FREE_BREAKFAST_BUILD_EXPERIMENTAL (the
# targets linked here exist only then) and NOT FREE_BREAKFAST_COMPILE_TIME_ONLY
# (GoogleTest exists only then). Its own binary, apart from test/gtest's two,
# so an experimental test can never break or slow the normal suite, and is
# dropped along with the code it tests.
#
# Registered exactly as test/gtest's GPU binary is -- one ctest entry per case,
# discovered PRE_TEST, labelled `gpu` -- for the reasons that file gives. Each
# case calls FBF_REQUIRE_GPU() and skips without a device.
include(GoogleTest)

add_executable(
  free_breakfast_tests_experimental_gpu
  radix_sort_gpu.cpp
  merge_sort_gpu.cpp
  fp16_gemm_gpu.cpp
  output_ring_gpu.cpp
  routine_gpu.cpp
)

set_target_properties(
  free_breakfast_tests_experimental_gpu PROPERTIES CXX_MODULE_STD 1
                                                   CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_tests_experimental_gpu
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::experimental::sort::radix
          free_breakfast::experimental::sort::merge
          free_breakfast::experimental::fp16_gemm
          free_breakfast::experimental::output_ring
          free_breakfast::experimental::routine
          free_breakfast::molecule
          free_breakfast::basis
          free_breakfast::recipes
          free_breakfast::integrals
          free_breakfast::integrals::eri::density_fitting::fit
          free_breakfast::response
          free_breakfast::mcscf
          free_breakfast::io
          free_breakfast::io::base
          free_breakfast::io::npy
          wwr::fp16
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

# routine_gpu.cpp reads test/gtest's golden data (the casscf_hessian
# fixtures), by the absolute path test/gtest gives its own binaries.
target_compile_definitions(
  free_breakfast_tests_experimental_gpu
  PRIVATE FBF_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/gtest/data"
)

math(EXPR _fb_test_timeout "120 * ${FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER}")

# Two labels: `gpu`, which the ci-* test presets exclude, and `experimental`,
# so `ctest -L experimental` runs exactly these. The `;` is escaped because
# PROPERTIES is a flat key/value list: unescaped, `experimental` is read as the
# next property's name and silently dropped.
gtest_discover_tests(
  free_breakfast_tests_experimental_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS "gpu\\;experimental"
)

# ---------------------------------------------------------------------------
# Must-not-compile entries of experimental/routine (#1246, #1324)
# ---------------------------------------------------------------------------
# Each TU is an OBJECT library excluded from `all`, and a ctest entry builds it
# and passes only on the diagnostic it should raise: a plain build failure
# would pass a TU broken for any reason, so the exit code is not what is
# checked. PASS_REGULAR_EXPRESSION makes ctest ignore it.
#
# The entries share one RESOURCE_LOCK: each runs a build in this build tree,
# and two builds in one Ninja tree at once race on .ninja_log and .ninja_deps.
# Labelled `experimental` only, not `gpu`: they need a compiler, not a device.
function(_fb_routine_must_not_compile name regex)
  set(target free_breakfast_routine_reject_${name})
  add_library(${target} OBJECT EXCLUDE_FROM_ALL routine_reject_${name}.cpp)
  set_target_properties(
    ${target} PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
  )
  target_link_libraries(
    ${target} PRIVATE free_breakfast::core free_breakfast::mcscf
                      free_breakfast::experimental::routine
  )
  add_test(NAME RoutineMustNotCompile.${name}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
                   ${target}
  )
  set_tests_properties(
    RoutineMustNotCompile.${name}
    PROPERTIES PASS_REGULAR_EXPRESSION "${regex}"
               RESOURCE_LOCK free_breakfast_build_tree
               LABELS experimental
               TIMEOUT ${_fb_test_timeout}
  )
endfunction()

# (a) An Alloc::No routine calls an Alloc::Yes one: it has no key to pass.
_fb_routine_must_not_compile(
  alloc_up "requires 2 arguments, but 1 was provided"
)
# (b) An Alloc::No routine constructs the keyed buffer: AllocKey{} is private.
_fb_routine_must_not_compile(
  key_buffer "calling a private constructor of class '.*AllocKey'"
)
# (c) The key forged with std::bit_cast: AllocKey is not trivially copyable.
_fb_routine_must_not_compile(bit_cast "is_trivially_copyable")
# (d) A misspelled In field at a call site. clang 23 does not name the field
# here: a braced list passed to any function parameter fails as overload
# resolution, "cannot convert initializer list argument", for a plain function
# as for the routine's template (experimental/routine/README.md). The regex
# pins the In conversion, the narrowest text the diagnostic carries.
string(CONCAT _fb_misspelled_regex
              "cannot convert initializer list argument to 'const typename "
              "free_breakfast::routine::RootRotationRoutine::In'"
)
_fb_routine_must_not_compile(misspelled_field "${_fb_misspelled_regex}")
# (e) A key forged outside the root's module (#1324): a TU defines a class
# named for AllocKey's friend. The friend is attached to the key's module, so
# the definition is refused before the constructor is reached.
_fb_routine_must_not_compile(
  forged_root "declaration of 'RootRotationStep' in the global module"
)
