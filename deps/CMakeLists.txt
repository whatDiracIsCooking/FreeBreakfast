# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Four kinds of thing live here, and they are wired differently on purpose:
#
#   WarpWraps      a git submodule, built from source as part of this build.
#                  Supplies the GPU API layer -- and the GoogleTest the test
#                  tier links.
#   spdlog         find_package. Prebuilt into /opt by docker/Dockerfile.base,
#   taskflow       because spdlog is COMPILED and so has to agree with this
#   tomlplusplus   tree about the standard library (libc++, not libstdc++).
#   nlohmann_json  FetchContent: header-only and small enough not to be worth a
#                  prebuild.
#
# Each of the four non-GPU libraries gets a thin `.cppm` module shim in its own
# subdirectory, so the rest of the tree says
# `import free_breakfast.third_party.spdlog` rather than including a header
# inside a global module fragment.
include(FetchContent)

# ==========================================================================
# WarpWraps (wwr) -- the CUDA/HIP abstraction layer
# ==========================================================================
# A submodule rather than a FetchContent declaration: the pin is then visible in
# `git submodule status` and reviewable in a diff, and a checkout can be edited
# in place when a change has to be made on both sides of the boundary at once.
#
# Fail here, naming the submodule, rather than letting add_subdirectory report
# a missing CMakeLists.txt -- which names this file and not the thing that is
# actually absent, and is a confusing first experience in a fresh clone.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/WarpWraps/CMakeLists.txt")
  message(
    FATAL_ERROR
      "deps/WarpWraps is empty -- the submodule has not been initialised. "
      "Run `git submodule update --init --recursive` and configure again."
  )
endif()

# The wrappers for libraries that are NOT part of the base CUDA/ROCm toolkit.
# OFF, which is wwr's own default but NOT what its dev presets use: left ON,
# the build would require nvCOMP and cuTENSOR (and NCCL, or RCCL on the HIP
# side) to be present, and docker/Dockerfile.cuda would have to install all
# three. Turn one back ON here when this project actually imports it, and add
# the library to the image in the same change.
set(WWR_WITH_CCL
    OFF
    CACHE BOOL "" FORCE
)
set(WWR_WITH_COMP
    OFF
    CACHE BOOL "" FORCE
)
set(WWR_WITH_TENSOR
    OFF
    CACHE BOOL "" FORCE
)

# What of wwr follows this project into ITS install tree: the core wwr* modules
# (always shipped once WWR_INSTALL is ON) plus the wrappers layer, and NOT the
# extension layer. wwr defaults WWR_INSTALL to PROJECT_IS_TOP_LEVEL -- OFF for a
# nested build -- so it has to be turned on here; the two layer switches are set
# explicitly both ways so this does not depend on wwr's defaults staying put.
#
# The extension is still BUILT (src/core links wwr::extension::*); this only
# keeps it out of the installed package. The catch: an install(EXPORT) of a
# free_breakfast target that links wwr::extension::* would be refused, because
# its link targets are not in any export set. That is moot while this project
# has no install rules of its own, and is the line to revisit when it does.
set(WWR_INSTALL
    ON
    CACHE BOOL "" FORCE
)
set(WWR_INSTALL_WRAPPERS
    ON
    CACHE BOOL "" FORCE
)
set(WWR_INSTALL_EXTENSION
    OFF
    CACHE BOOL "" FORCE
)

# wwr's own test suites and examples. Both already default to OFF for a nested
# build (PROJECT_IS_TOP_LEVEL), and are set explicitly so this does not depend
# on a default staying put. OFF keeps wwr's ctest entries out of this
# project's run and its examples out of `all`; wwr's suites are verified by
# wwr's own CI, not here. GoogleTest is unaffected -- wwr still fetches it,
# see below.
set(WWR_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE
)
set(WWR_BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE
)

# WWR_GPU_BACKEND is set from FREE_BREAKFAST_GPU_BACKEND in the root
# CMakeLists, before project() -- it has to be, because it decides whether wwr
# enables the CUDA language. Do not set it here.
add_subdirectory(WarpWraps)

# ==========================================================================
# FelizJueves -- the kernel generator's headers
# ==========================================================================
# A submodule, and NOT built: it is a Python generator, run by hand
# (tools/emit_one_electron_kernels.py), whose output is committed under src/.
# What the build needs of it is its include/ -- felizjueves/runtime.h, which
# every kernel its `gpu` backend writes includes, and which selects the vendor
# runtime from the compiler's own macros, so it needs nothing of wwr.
#
# FJ_WAVEFRONT is what each generated unit with a shuffle static-asserts its
# width against. It is wwr's warp size by definition -- one tree, one target
# arch -- so it is mirrored from WWR_WARP_SIZE here rather than set anywhere
# else.
set(_felizjueves_include "${CMAKE_CURRENT_SOURCE_DIR}/FelizJueves/include")
if(NOT EXISTS "${_felizjueves_include}/felizjueves/runtime.h")
  message(
    FATAL_ERROR
      "deps/FelizJueves is empty or predates its include/ -- the submodule "
      "has not been initialised, or is pinned too old. Run "
      "`git submodule update --init --recursive` and configure again."
  )
endif()
add_library(felizjueves_headers INTERFACE)
add_library(felizjueves::headers ALIAS felizjueves_headers)
target_include_directories(
  felizjueves_headers INTERFACE "${_felizjueves_include}"
)
target_compile_definitions(
  felizjueves_headers INTERFACE FJ_WAVEFRONT=${WWR_WARP_SIZE}
)

# ==========================================================================
# GoogleTest
# ==========================================================================
# NOT declared here. wwr's own deps/CMakeLists.txt fetches it (v1.17.0, stock,
# with CXX_SCAN_FOR_MODULES OFF) and that runs as part of the add_subdirectory
# above, so the GTest::* targets already exist by the time test/ is added.
#
# Declaring a second googletest would not give this project a different one --
# FetchContent is first-declaration-wins, and wwr declares first -- it would
# only make the pin look like it lived here.
#
# This is also why there is no googletest-std-module.patch in this tree. The
# approach it came from built GoogleTest with GTEST_BUILD_USING_STD_MODULE so
# its sources would `import std;` like the rest of the tree. wwr instead
# declares GoogleTest's TUs plain C++ and skips module scanning on them, which
# reaches the same place -- consuming TUs get the ordinary header build either
# way, since that define was always PRIVATE to GoogleTest's own sources -- with
# no patch to re-roll on every upgrade.
if(NOT FREE_BREAKFAST_COMPILE_TIME_ONLY AND NOT TARGET gtest)
  message(
    FATAL_ERROR
      "GoogleTest was not defined by deps/WarpWraps, and this is not a "
      "FREE_BREAKFAST_COMPILE_TIME_ONLY configure. It is expected to come from "
      "wwr's own deps/, which wwr skips when WWR_COMPILE_TIME_ONLY is ON."
  )
endif()

# ==========================================================================
# The non-GPU libraries
# ==========================================================================
find_package(spdlog REQUIRED)
find_package(Taskflow REQUIRED)
find_package(tomlplusplus REQUIRED)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory(nlohmann)
add_subdirectory(spdlog)
add_subdirectory(taskflow)
add_subdirectory(tomlplusplus)
