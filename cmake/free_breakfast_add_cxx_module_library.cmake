# FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY.cmake Provides
# FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY macro for creating C++23 module
# libraries

# ---------------------------------------------------------------------------
# Internal helpers (underscore-prefixed — not part of the public API)
# ---------------------------------------------------------------------------

# Add the selected GPU backend's toolkit include directories, with validated
# visibility. SYSTEM, so vendor headers never produce warnings this project is
# expected to fix.
#
# Backend-aware rather than CUDA-only: CUDAToolkit_INCLUDE_DIRS does not exist
# in a HIP build (no find_package(CUDAToolkit) runs), so naming it
# unconditionally expanded to nothing there and the include silently went
# missing. The HIP side reads the same directories off hip::host -- hip::host
# and not hip::device, because this is a HOST module compile and hip::device
# would drag `-x hip` onto it.
function(_free_breakfast_configure_gpu_toolkit target visibility)
  if(NOT visibility)
    set(visibility "PRIVATE")
  elseif(NOT visibility STREQUAL "PUBLIC" AND NOT visibility STREQUAL "PRIVATE")
    message(FATAL_ERROR "FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY:"
                        " INCLUDE_GPU_TOOLKIT must be PUBLIC or PRIVATE"
    )
  endif()
  if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
    set(gpu_inc "${CUDAToolkit_INCLUDE_DIRS}")
  else()
    get_target_property(gpu_inc hip::host INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  target_include_directories(${target} SYSTEM ${visibility} ${gpu_inc})
endfunction()

# ---------------------------------------------------------------------------
# Public macro
# ---------------------------------------------------------------------------

# Macro to create a C++23 module library with standard boilerplate Usage:
# FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY( NAME library_name PRIMARY_INTERFACE
# file.cppm [PARTITIONS partition1.cppm partition2.cppm ...] [IMPLEMENTATION
# impl1.cpp impl2.cpp ...] [LINK_PUBLIC lib1 lib2 ...] [LINK_PRIVATE lib3 lib4
# ...] [IMPORT_STD] [NO_GPU_DEVICE_LINKING] [INCLUDE_GPU_TOOLKIT
# PUBLIC|PRIVATE] )
#
# Parameters:
#   PRIMARY_INTERFACE     the single primary module interface file
#                         (export module M;)
#   PARTITIONS            partition files of BOTH kinds: interface partitions
#                         (export module M:P;) and internal partitions
#                         (module M:P;). Both produce a BMI, so both belong in
#                         the CXX_MODULES file set; docs/style.md says which
#                         kind module-private code goes in.
#   IMPLEMENTATION        module implementation units (module M;)
#   IMPORT_STD            flag: enable 'import std' (sets CXX_MODULE_STD 1)
#   NO_GPU_DEVICE_LINKING flag: CUDA_SEPARABLE_COMPILATION and
#                         CUDA_RESOLVE_DEVICE_SYMBOLS OFF (CUDA backend only;
#                         inert under HIP)
#   INCLUDE_GPU_TOOLKIT   PUBLIC|PRIVATE (default PRIVATE): add the selected
#                         backend's toolkit include dirs as SYSTEM includes --
#                         CUDAToolkit_INCLUDE_DIRS under CUDA, hip::host's
#                         under HIP
#   INCLUDE_DIRS_PUBLIC   directories to add as PUBLIC include paths
#   INCLUDE_DIRS_PRIVATE  directories to add as PRIVATE include paths
macro(FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY)
  set(_bool_opts "IMPORT_STD;NO_GPU_DEVICE_LINKING")
  set(_single_opts "NAME;PRIMARY_INTERFACE;INCLUDE_GPU_TOOLKIT;WHOLE_ARCHIVE")
  set(_multi_opts "PARTITIONS;IMPLEMENTATION;LINK_PUBLIC;LINK_PRIVATE"
                  ";INCLUDE_DIRS_PUBLIC;INCLUDE_DIRS_PRIVATE"
  )
  cmake_parse_arguments(
    ARG "${_bool_opts}" "${_single_opts}" "${_multi_opts}" ${ARGN}
  )

  if(NOT ARG_NAME)
    message(
      FATAL_ERROR "FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY: NAME is required"
    )
  endif()
  if(NOT ARG_PRIMARY_INTERFACE)
    message(
      FATAL_ERROR
        "FREE_BREAKFAST_ADD_CXX_MODULE_LIBRARY: PRIMARY_INTERFACE is required"
    )
  endif()

  add_library(${ARG_NAME})

  set(ALL_MODULE_INTERFACE_FILES ${ARG_PRIMARY_INTERFACE})
  if(ARG_PARTITIONS)
    list(APPEND ALL_MODULE_INTERFACE_FILES ${ARG_PARTITIONS})
  endif()

  target_sources(
    ${ARG_NAME} PUBLIC FILE_SET CXX_MODULES FILES ${ALL_MODULE_INTERFACE_FILES}
  )

  if(ARG_IMPLEMENTATION)
    target_sources(${ARG_NAME} PRIVATE ${ARG_IMPLEMENTATION})
  endif()

  if(ARG_IMPORT_STD)
    set_target_properties(${ARG_NAME} PROPERTIES CXX_MODULE_STD 1)
  endif()

  # CUDA-only: the CUDA_* properties are inert in a HIP build, which enables no
  # CUDA language and never turns separable compilation on in the first place.
  if(ARG_NO_GPU_DEVICE_LINKING AND FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
    set_target_properties(
      ${ARG_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
  endif()

  if(DEFINED ARG_INCLUDE_GPU_TOOLKIT)
    _free_breakfast_configure_gpu_toolkit(
      ${ARG_NAME} "${ARG_INCLUDE_GPU_TOOLKIT}"
    )
  endif()

  if(ARG_INCLUDE_DIRS_PUBLIC OR ARG_INCLUDE_DIRS_PRIVATE)
    _free_breakfast_configure_include_dirs(
      ${ARG_NAME} PUBLIC ${ARG_INCLUDE_DIRS_PUBLIC} PRIVATE
      ${ARG_INCLUDE_DIRS_PRIVATE}
    )
  endif()

  if(ARG_LINK_PUBLIC OR ARG_LINK_PRIVATE)
    _free_breakfast_configure_link_libraries(
      ${ARG_NAME} PUBLIC ${ARG_LINK_PUBLIC} PRIVATE ${ARG_LINK_PRIVATE}
    )
  endif()

  if(ARG_WHOLE_ARCHIVE)
    target_link_options(
      ${ARG_NAME} INTERFACE "-Wl,--whole-archive"
      "$<TARGET_FILE:${ARG_WHOLE_ARCHIVE}>" "-Wl,--no-whole-archive"
    )
  endif()

  _free_breakfast_create_alias(${ARG_NAME})

endmacro()
