cmake_minimum_required(VERSION 4.2)

# `import std` is still gated behind an experimental feature flag, and CMake
# accepts a different gate UUID per release ON PURPOSE, to force
# re-acknowledgement of the unstable API on each upgrade. The accepted value
# lives only inside the cmake binary -- it cannot be derived at configure time
# -- so it must be pinned to the version it belongs to. The failure mode is
# silent: a stale UUID is IGNORED, and every `import std` then fails to resolve.
# So gate the UUID on the CMake version and hard-error on any version whose
# value has not been verified here, rather than let an upgrade quietly break
# module resolution.
#
# When upgrading CMake, find that release's value (Help/dev/experimental.rst, or
# `strings $(command -v cmake)` and match the UUID pattern), confirm it, and add
# a branch below. Verified values:
#   4.2.x  ->  d0edc3af-4c50-42ea-a356-e2862fe7a444   (confirmed against 4.2.3)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.2 AND CMAKE_VERSION VERSION_LESS 4.3)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
else()
  message(
    FATAL_ERROR
      "No verified CMAKE_EXPERIMENTAL_CXX_IMPORT_STD gate UUID for CMake "
      "${CMAKE_VERSION}. The UUID is release-specific and a stale one is "
      "ignored silently, breaking every `import std`. Look up this release's "
      "value (Help/dev/experimental.rst, or `strings $(command -v cmake)`), "
      "verify it, and add a branch above."
  )
endif()

# Configure libc++ BEFORE project() so compiler detection uses it.
# CMAKE_CXX_STANDARD_LIBRARY must be set explicitly so CMake looks for
# libc++.modules.json (not libstdc++.modules.json) when resolving import std.
set(CMAKE_CXX_STANDARD_LIBRARY "libc++")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")

# Set the C++ compiler to Clang BEFORE project(). Resolve to a full path: a bare
# name is re-resolved on every configure, and once a nested project() (the
# WarpWraps subdirectory) has cached CMAKE_CXX_COMPILER_WORKS, a bare name is
# rejected as "not a full path and not found in the PATH". A full path is
# validated identically either way.
find_program(_fb_cxx_compiler NAMES clang++ REQUIRED)
set(CMAKE_CXX_COMPILER "${_fb_cxx_compiler}")

# ========================================================================
# Toolchain discovery (must run before project())
# ========================================================================
# C++23 modules need clang-scan-deps, and `import std` needs libc++'s module
# manifest. Debian/Ubuntu install both under a versioned /usr/lib/llvm-<N>/ that
# is not on PATH, so search there, preferring the newest version found. Override
# either if discovery picks the wrong toolchain:
#
#   -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=/path/to/clang-scan-deps
#   -DCMAKE_CXX_STDLIB_MODULES_JSON=/path/to/libc++.modules.json
file(
  GLOB _llvm_roots
  LIST_DIRECTORIES true
  /usr/lib/llvm-* /usr/local/llvm-*
)
list(
  SORT _llvm_roots
  COMPARE NATURAL
  ORDER DESCENDING
)

set(_llvm_bin_hints "")
set(_llvm_lib_hints "")
foreach(_root IN LISTS _llvm_roots)
  list(APPEND _llvm_bin_hints "${_root}/bin")
  list(APPEND _llvm_lib_hints "${_root}/lib")
endforeach()

if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  find_program(
    CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps
    HINTS ${_llvm_bin_hints}
  )
  if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    message(
      FATAL_ERROR
        "clang-scan-deps not found (required for C++23 module scanning). "
        "Install the matching LLVM tools package, or pass "
        "-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=<path>."
    )
  endif()
endif()

if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
  find_file(
    CMAKE_CXX_STDLIB_MODULES_JSON
    NAMES libc++.modules.json
    HINTS ${_llvm_lib_hints}
    PATHS /usr/lib /usr/local/lib
  )
  if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    message(
      FATAL_ERROR
        "libc++.modules.json not found (required for `import std`). "
        "Install libc++-dev built with module support, or pass "
        "-DCMAKE_CXX_STDLIB_MODULES_JSON=<path>."
    )
  endif()
endif()

message(STATUS "Using clang-scan-deps: ${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}")
message(
  STATUS "Using libc++ modules manifest: ${CMAKE_CXX_STDLIB_MODULES_JSON}"
)

# ========================================================================
# GPU backend
# ========================================================================
# A build targets exactly one GPU backend, CUDA or HIP. Decided before project()
# because it decides whether the CUDA language is enabled at all.
#
#   CUDA -> the CUDA toolkit, nvcc for `.cu`, deps/WarpWraps's src/cuda layer
#   HIP  -> ROCm, clang `-x hip` for `.cu`, deps/WarpWraps's src/hip layer
#
# Either way WarpWraps maps its backend-neutral `wwr*` names onto the chosen
# backend, so everything under src/ is written once and builds for both. A HIP
# build needs no CUDA toolkit, and a CUDA build needs no ROCm.
#
# No default on purpose: the value is read before project() to decide whether
# the CUDA language is enabled, so a silent default would quietly require a CUDA
# toolkit on a HIP-only host (or ROCm on a CUDA-only one). Every preset in
# CMakePresets.json carries it -- CUDA via the hidden `base` preset, HIP via
# `hip`/`ci-hip` -- so only a raw preset-less `cmake` has to pass it. On
# reconfigure the cached value satisfies this, so it need not be repassed.
if(NOT DEFINED FREE_BREAKFAST_GPU_BACKEND)
  message(
    FATAL_ERROR
      "FREE_BREAKFAST_GPU_BACKEND must be set explicitly to CUDA or HIP (e.g. "
      "-DFREE_BREAKFAST_GPU_BACKEND=CUDA, or a CMakePresets.json preset). It "
      "is not defaulted: it is read before project() to decide whether the "
      "CUDA language is enabled at all."
  )
endif()
set(FREE_BREAKFAST_GPU_BACKEND
    "${FREE_BREAKFAST_GPU_BACKEND}"
    CACHE STRING "GPU backend to build against: CUDA or HIP"
)
set_property(CACHE FREE_BREAKFAST_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT FREE_BREAKFAST_GPU_BACKEND MATCHES "^(CUDA|HIP)$")
  message(
    FATAL_ERROR "FREE_BREAKFAST_GPU_BACKEND must be CUDA or HIP, got "
                "'${FREE_BREAKFAST_GPU_BACKEND}'"
  )
endif()

# WarpWraps reads its own backend switch from WWR_GPU_BACKEND. Mirror this
# project's choice onto it here, before project() and before the subdirectory is
# added, so the two can never disagree -- a tree built CUDA against a HIP wwr
# fails deep inside wwr's own sources with nothing naming the mismatch.
set(WWR_GPU_BACKEND
    "${FREE_BREAKFAST_GPU_BACKEND}"
    CACHE STRING "WarpWraps GPU backend (follows FREE_BREAKFAST_GPU_BACKEND)"
          FORCE
)

# Configure only what can be proved by COMPILING -- no GoogleTest, no runtime
# test binaries, no device needed. This is what devtools/cross-backend-check.sh
# configures with when it compiles the tree for the backend this build is NOT
# targeting: that check only needs to know the code still compiles, and skipping
# the test tier makes it seconds rather than minutes.
#
# Declared up here, before project(), because the block just below reads it --
# an option() further down the file would be a declaration after its first use,
# which works only by accident of the command-line cache entry already existing.
option(
  FREE_BREAKFAST_COMPILE_TIME_ONLY
  "Configure only the compile-time checks; skip GoogleTest and the test tier"
  OFF
)

# Same mirroring for the compile-time-only switch, and for the same reason it
# has to happen here: wwr reads it to decide whether to add its own deps/ (the
# GoogleTest fetch), which runs during the add_subdirectory in deps/.
if(FREE_BREAKFAST_COMPILE_TIME_ONLY)
  set(WWR_COMPILE_TIME_ONLY
      ON
      CACHE BOOL "Follows FREE_BREAKFAST_COMPILE_TIME_ONLY" FORCE
  )
endif()

if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  set(_fb_languages CXX CUDA)
else()
  set(_fb_languages CXX)
endif()

project(
  free-breakfast
  VERSION 0.1.0
  LANGUAGES ${_fb_languages}
  DESCRIPTION
  "GPU-first scientific computing project in C++23, CUDA or HIP"
)
message(STATUS "GPU backend: ${FREE_BREAKFAST_GPU_BACKEND}")

# ========================================================================
# C++ Standard and Compiler Requirements
# ========================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Note: CMAKE_CXX_EXTENSIONS must match what the std module uses. Leave it at
# default (ON) to match libc++ std module compilation.
set(CMAKE_CXX_EXTENSIONS ON)

# Require Clang with libc++
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(
    FATAL_ERROR "This project requires Clang compiler for C++23 module support"
  )
endif()

# Add libc++ linker flags for CXX targets
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-lc++abi>)

# ========================================================================
# Find the backend's packages
# ========================================================================
if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit 13.0.0 REQUIRED)

  # The floor here is WarpWraps's, not the toolkit's: wwr wraps and tests bf16
  # tensor-core WMMA, an Ampere feature, so it rejects an explicit sub-80
  # CMAKE_CUDA_ARCHITECTURES at configure. CUDA 13's own nvcc floor is lower
  # (compute_75), but wwr's is the binding one -- which is why there is no
  # `volta` preset and why docker/Dockerfile.cuda's CUDA_ARCH stays >= 80.
  # Checked here as well as in wwr so the message names THIS project's
  # requirement at the point the architecture is chosen. `native` and the `all*`
  # keywords are resolved by nvcc rather than by this loop, so they never match
  # and pass through; an `NN-real`/`NN-virtual` suffix keeps its leading digits.
  foreach(_fb_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_fb_arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
      message(
        FATAL_ERROR
          "free-breakfast requires CUDA architecture 80 (Ampere) or newer -- "
          "its WarpWraps dependency wraps bf16 WMMA -- got '${_fb_arch}' in "
          "CMAKE_CUDA_ARCHITECTURES"
      )
    endif()
  endforeach()

  # nvcc links libstdc++ internally. When the host linker is clang with
  # -stdlib=libc++, the GCC library directory is not in the default search path
  # and -lstdc++ fails to resolve. Expose it via link_directories and rpath.
  execute_process(
    COMMAND gcc --print-file-name=libstdc++.so
    OUTPUT_VARIABLE STDCXX_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _gcc_result
  )
  if(_gcc_result EQUAL 0 AND NOT STDCXX_LIBRARY STREQUAL "libstdc++.so")
    get_filename_component(STDCXX_DIR "${STDCXX_LIBRARY}" DIRECTORY)
    add_link_options("-L${STDCXX_DIR}")
    list(APPEND CMAKE_BUILD_RPATH "${STDCXX_DIR}")
    message(STATUS "Found libstdc++ directory: ${STDCXX_DIR}")
  endif()
else()
  # Seed GPU_TARGETS from the environment before find_package(hip).
  #
  # hip-config-amd.cmake reads the CMake variable GPU_TARGETS; the HIP images
  # export an env var of the same name, but an env var is not a CMake variable,
  # so it must be copied in. Left unset, hip-config runs `amdgpu-arch` and on a
  # box with no AMD GPU silently falls back to gfx906 -- wave64, which also
  # disagrees with the warp size WarpWraps defaults to. Precedence: an explicit
  # -DGPU_TARGETS (already cached) beats $ENV{GPU_TARGETS} beats ROCm's own
  # detection.
  if(NOT DEFINED GPU_TARGETS AND DEFINED ENV{GPU_TARGETS})
    set(GPU_TARGETS
        "$ENV{GPU_TARGETS}"
        CACHE STRING "AMD GPU targets to compile for (from the environment)"
    )
    message(STATUS "GPU_TARGETS from environment: ${GPU_TARGETS}")
  endif()

  find_package(hip CONFIG REQUIRED)
endif()

# ========================================================================
# Compiler cache
# ========================================================================
# Wired in by find_program here rather than by a preset, so an image or host
# without sccache still builds -- it just caches nothing. It caches clang TUs,
# including the C++23 module interface units (compiled with -fmodule-output /
# -fmodule-file=NAME=PATH, both in sccache's supported set), and nvcc device
# TUs.
#
# Device-pass `.cu` sources go through devtools/gpu-object-cache.sh rather
# than sccache directly: it keys on content, not on the worktree path sccache
# sees, so an identical kernel compiled in another worktree is a hit. See that
# script's header. Under CUDA every `.cu` is the CUDA language, so the
# language-wide launcher covers exactly them. Under HIP a `.cu` is CXX, and the
# CXX launcher also fronts the module units, whose BMI the object cache cannot
# restore -- so FREE_BREAKFAST_GPU_OBJECT_CACHE is attached per target, by
# free_breakfast_add_gpu_device_library, and the salt carries the ROCm version
# that apt's clang --version does not.
find_program(FREE_BREAKFAST_SCCACHE sccache)
if(FREE_BREAKFAST_SCCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER "${FREE_BREAKFAST_SCCACHE}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${FREE_BREAKFAST_SCCACHE}")
  set(_fbf_gpu_object_cache
      "${CMAKE_CURRENT_SOURCE_DIR}/devtools/gpu-object-cache.sh"
  )
  if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_fbf_gpu_object_cache}")
  else()
    set(FREE_BREAKFAST_GPU_OBJECT_CACHE
        "${_fbf_gpu_object_cache};--salt=hip-${hip_VERSION}"
    )
  endif()
  unset(_fbf_gpu_object_cache)
  message(STATUS "Compiler cache: ${FREE_BREAKFAST_SCCACHE}")
else()
  message(
    STATUS "Compiler cache: none (sccache not found; builds are uncached)"
  )
endif()

# ========================================================================
# Module Configuration
# ========================================================================
# The macros that define every target in this project. See cmake/README.md.
include(cmake/free_breakfast_internal_helpers.cmake)
include(cmake/free_breakfast_add_cxx_module_library.cmake)
include(cmake/free_breakfast_add_gpu_device_library.cmake)
include(cmake/free_breakfast_add_interface_library.cmake)

# ========================================================================
# Sanitizers (host code only)
# ========================================================================
option(FREE_BREAKFAST_ENABLE_ASAN
       "Enable AddressSanitizer instrumentation for host (CXX) code" OFF
)
if(FREE_BREAKFAST_ENABLE_ASAN)
  message(STATUS "AddressSanitizer enabled (host code only)")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
  )
  add_link_options(-fsanitize=address)
endif()

# ========================================================================
# Coverage (host code only)
# ========================================================================
# clang source-based coverage, driven by devtools/coverage.sh and turned on by
# the `coverage` preset. Host CXX only: a device pass has no counterpart for
# these flags, and adding them to a `.cu` compile fails outright.
option(FREE_BREAKFAST_COVERAGE
       "Instrument host (CXX) code for clang source-based coverage" OFF
)
if(FREE_BREAKFAST_COVERAGE)
  message(STATUS "Coverage instrumentation enabled (host code only)")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fprofile-instr-generate>
    $<$<COMPILE_LANGUAGE:CXX>:-fcoverage-mapping>
  )
  # The link options are NOT language-gated: the instrumented objects need
  # clang's profile runtime linked into the final binary, and a generator
  # expression on COMPILE_LANGUAGE does not apply at link time.
  add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

# ========================================================================
# Options
# ========================================================================
# ========================================================================
# compute-sanitizer
# ========================================================================
# When non-empty, every ctest entry runs under NVIDIA compute-sanitizer with
# this tool (memcheck / racecheck / initcheck / synccheck). Empty -- the default
# -- leaves ctest running the binaries directly.
#
# This is a CMAKE_TEST_LAUNCHER rather than a prefix baked into each add_test,
# so it covers every test in the tree (the WarpWraps submodule's included)
# without any of them knowing about it. It has to be set BEFORE the test targets
# are created, because CMake snapshots it onto each target as TEST_LAUNCHER at
# creation time -- which is why this sits up here and not beside
# add_subdirectory(test).
#
# docker/compose.yaml's `compute-sanitizer` service passes this at configure
# (-DFREE_BREAKFAST_COMPUTE_SANITIZER_TOOL="$tool"). Without the option, that -D
# was accepted and ignored: the service built, ran plain ctest, and reported
# success having checked nothing -- a green run that proved the opposite of what
# it claimed.
#
# TWO flags make this usable, and both were learned the hard way:
#
#   --error-exitcode=1   compute-sanitizer exits 0 by default even when it
#                        reports errors, so without this a memcheck finding
#                        prints and the test still passes.
#   --require-cuda-init no
#                        ...and WITH it, every binary that never calls a CUDA
#                        API fails. The tool treats "terminated before first
#                        instrumented API call" as an error, and the launcher
#                        wraps more than the test bodies: gtest_discover_tests
#                        runs `--gtest_list_tests` through it at ctest time
#                        (DISCOVERY_MODE PRE_TEST), and that invocation touches
#                        no device at all. The result was `Result: 255` during
#                        DISCOVERY, which fails the whole ctest run rather than
#                        one test, and reads as a CMake error about
#                        gtest_discover_tests_impl rather than anything to do
#                        with the sanitizer. The host-only test binary hits the
#                        same thing legitimately.
#                        docker/compose.yaml passes the same flag, for the same
#                        reason, when it runs pytest under the tool.
set(FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL
    ""
    CACHE STRING
    "Run every ctest entry under compute-sanitizer with this tool (empty = off)"
)
set_property(
  CACHE FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL
  PROPERTY STRINGS
           ""
           memcheck
           racecheck
           initcheck
           synccheck
)
if(FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL)
  if(NOT FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
    message(
      FATAL_ERROR
        "FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL is CUDA-only; this is a "
        "${FREE_BREAKFAST_GPU_BACKEND} build. ROCm's equivalent is a different "
        "tool with a different invocation."
    )
  endif()
  find_program(
    FREE_BREAKFAST_COMPUTE_SANITIZER
    NAMES compute-sanitizer
    HINTS "${CUDAToolkit_BIN_DIR}" /usr/local/cuda/bin
    REQUIRED
  )
  set(CMAKE_TEST_LAUNCHER
      "${FREE_BREAKFAST_COMPUTE_SANITIZER}"
      "--tool=${FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL}"
      "--error-exitcode=1"
      "--require-cuda-init"
      "no"
  )
  message(
    STATUS
      "compute-sanitizer: every test runs under "
      "--tool=${FREE_BREAKFAST_COMPUTE_SANITIZER_TOOL}"
  )
endif()

# ========================================================================
# Test timeouts
# ========================================================================
# Multiplies every TIMEOUT this project sets on a ctest entry (see
# test/CMakeLists.txt). 1 is the default and changes nothing.
#
# It exists for the sanitizer runs: compute-sanitizer slows a GPU test by an
# order of magnitude, so the timeout that is a generous backstop for a normal
# run becomes the thing that fails the suite. docker/compose.yaml's
# compute-sanitizer service passes it (its own default is 10) -- and, like the
# option above, that -D was accepted and ignored before this existed.
set(FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER
    "1"
    CACHE STRING "Scale every ctest TIMEOUT this project sets"
)
if(NOT FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER MATCHES "^[1-9][0-9]*$")
  message(
    FATAL_ERROR
      "FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER must be a positive integer, got "
      "'${FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER}'"
  )
endif()

option(FREE_BREAKFAST_BUILD_EXPERIMENTAL "Build experimental modules and tests"
       OFF
)

option(FREE_BREAKFAST_BUILD_BENCH
       "Build bench/ wall-clock benchmarks (drive the real modules; opt-in)" OFF
)

# ========================================================================
# Testing
# ========================================================================
# BEFORE any add_subdirectory, and that position is load-bearing. CMake records
# a `subdirs(...)` line in a directory's CTestTestfile.cmake only if testing was
# enabled when that directory was PROCESSED, so add_test() in a directory added
# above this call is silently dropped -- no error, just fewer tests and a green
# run. It is the same trap the experimental/ guard at the bottom of this file
# calls out; keeping the call first means every directory gets the same
# treatment regardless of where it sits.
#
# (WarpWraps' own suites are NOT part of this: deps/CMakeLists.txt turns them
# off with WWR_BUILD_TESTS.)
enable_testing()

# ========================================================================
# Dependencies, then project source
# ========================================================================
add_subdirectory(deps)
add_subdirectory(src)

# The C++ generators tools/ runs by hand (tools/README.md). Built with the
# tree, so a generator cannot rot unbuilt between regenerations.
add_subdirectory(tools/sad)

# The device kernels the core GPU tests drive. Outside the gate below on
# purpose: they need no GoogleTest, and devtools/cross-backend-check.sh (which
# configures with FREE_BREAKFAST_COMPILE_TIME_ONLY) exists precisely to compile
# .cu code for the other backend.
add_subdirectory(test/core/gpu_kernels)

# Skipped under FREE_BREAKFAST_COMPILE_TIME_ONLY: there is no GoogleTest to
# link against in that configuration (see the option's comment).
if(NOT FREE_BREAKFAST_COMPILE_TIME_ONLY)
  add_subdirectory(test)
endif()

# Standalone benchmarks that drive the real modules. Opt-in so they never add to
# the normal build: configure with -DFREE_BREAKFAST_BUILD_BENCH=ON.
if(FREE_BREAKFAST_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# experimental/ registers its own ctests, so it must be added *after*
# enable_testing() -- add_test() in a directory processed before that call is
# silently dropped (no CTestTestfile entry). Guarded, so it configures only
# under the `experimental` preset.
#
# test/experimental -- the tests of what experimental/ builds -- sits behind
# both guards: experimental's, because the targets it links exist only under
# it, and test/'s, because it needs GoogleTest, which a
# FREE_BREAKFAST_COMPILE_TIME_ONLY configure does not fetch.
if(FREE_BREAKFAST_BUILD_EXPERIMENTAL)
  add_subdirectory(experimental)
  if(NOT FREE_BREAKFAST_COMPILE_TIME_ONLY)
    add_subdirectory(test/experimental)
  endif()
endif()
