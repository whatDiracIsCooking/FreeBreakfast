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

# wwr defaults WWR_INSTALL to PROJECT_IS_TOP_LEVEL, which is already OFF for a
# nested build. Set explicitly anyway: it is the line that keeps wwr's headers
# and module sources from following this project into ITS install tree, and it
# should not depend on a default staying put.
set(WWR_INSTALL
    OFF
    CACHE BOOL "" FORCE
)

# WWR_GPU_BACKEND is set from FREE_BREAKFAST_GPU_BACKEND in the root
# CMakeLists, before project() -- it has to be, because it decides whether wwr
# enables the CUDA language. Do not set it here.
#
# NOTE ON WHAT THIS add_subdirectory BRINGS WITH IT: wwr's root CMakeLists adds
# its own example/ and test directories unconditionally and calls
# enable_testing() itself. There is no "skip my tests" switch, so a nested wwr
# builds and registers its own suites alongside this project's -- 78 ctest
# entries against this project's 4. That is accepted rather than worked around:
# the build cost is paid either way (there is no switch), so the suites may as
# well be runnable, and they are free verification that the submodule pin is
# sound on this toolchain. Three consequences to know about:
#
#   * `ctest` at the top level runs wwr's entries too. Filter with the labels
#     (`ctest -L compile_time`, `-LE gpu`) or by name when that is not what you
#     want.
#   * coverage over this project has to exclude them, which is what
#     COVERAGE_IGNORE_REGEX=/deps/ in devtools/config.sh does.
#   * it only works because enable_testing() is called BEFORE this
#     add_subdirectory, up in the root CMakeLists. Move it back down and
#     build/deps/CTestTestfile.cmake loses its subdirs("WarpWraps") line: wwr's
#     suites still COMPILE and silently become unrunnable, and the top-level run
#     reports 4 tests and passes. See that call's comment.
add_subdirectory(WarpWraps)

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
