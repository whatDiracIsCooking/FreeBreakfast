# libfbx_ssperi3_kernels.so: FelizJueves's fully spherical packed
# three-center kernels, ssperi3<la, lb, lP> (#745), and their launcher
# (ssperi3_launch.cu), for the df-batched-spherical crate beside this
# directory. A CMake project of its own, CUDA only, built neither by the root
# CMakeLists.txt nor by rust/'s:
#
#   cmake -S kernels -B kernels/build -G Ninja
#   cmake --build kernels/build
#
# The units are emitted at build time by emit.py from deps/FelizJueves, so
# nothing generated is committed. They include FelizJueves's runtime.h and the
# device Boys routine rust/integrals/kernels/shared/ carries; the launcher
# includes rust/'s class view header, the struct the Rust side hands it.
cmake_minimum_required(VERSION 3.28)
project(fbx_ssperi3 LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
cmake_path(NORMAL_PATH _root)
set(_fj_include "${_root}/deps/FelizJueves/include")
if(NOT EXISTS "${_fj_include}/felizjueves/runtime.h")
  message(FATAL_ERROR "deps/FelizJueves is empty: git submodule update --init")
endif()

# The 70 classes eri3_classes() lists: la <= lb <= 3, lP <= 6.
set(_gen "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(_units "")
foreach(_la IN ITEMS 0 1 2 3)
  foreach(_lb IN ITEMS 0 1 2 3)
    if(_lb LESS _la)
      continue()
    endif()
    foreach(_lp IN ITEMS 0 1 2 3 4 5 6)
      list(APPEND _units "${_gen}/ssperi3_${_la}${_lb}${_lp}.cu")
    endforeach()
  endforeach()
endforeach()
file(GLOB_RECURSE _fj_sources CONFIGURE_DEPENDS
     "${_root}/deps/FelizJueves/src/felizjueves/*.py"
)
add_custom_command(
  OUTPUT ${_units} "${_gen}/ssperi3.h"
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/emit.py" "${_gen}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/emit.py" ${_fj_sources}
  COMMENT "Emitting ssperi3 from deps/FelizJueves"
  VERBATIM
)

add_library(fbx_ssperi3_kernels SHARED ssperi3_launch.cu ${_units})
# rust/'s device headers (kFourWarps, math::idivup, the launch statuses),
# over WarpWraps' neutral runtime.h, read from the submodule's source tree:
# headers only, so no wwr target is built. The warp width is CUDA's, set
# once for both FelizJueves and wwr.
target_include_directories(
  fbx_ssperi3_kernels
  PRIVATE "${_gen}" "${_fj_include}"
          "${_root}/rust/integrals/kernels/shared"
          "${_root}/rust/integrals/kernels"
          "${_root}/rust/device_headers"
          "${_root}/deps/WarpWraps/src"
)
target_compile_definitions(
  fbx_ssperi3_kernels PRIVATE FJ_WAVEFRONT=32 WWR_WARP_SIZE=32
)
# The launcher launches specializations the units define: relocatable device
# code, the device link resolved inside the .so, as rust/'s kernel libraries.
set_target_properties(
  fbx_ssperi3_kernels
  PROPERTIES CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_RUNTIME_LIBRARY Shared
             POSITION_INDEPENDENT_CODE ON
)
