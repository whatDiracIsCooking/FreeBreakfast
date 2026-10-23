# FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY.cmake Provides
# FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY for a STATIC library of device-compiled
# `.cu` sources, built for whichever backend FREE_BREAKFAST_GPU_BACKEND
# selected.
#
# This replaced an earlier FREE_BREAKFAST_ADD_CUDA_LIBRARY, which hardcoded the
# CUDA language and CUDAToolkit_INCLUDE_DIRS and so could only ever build one
# backend. `.cu` here means "device pass", not "nvcc": under HIP the file is
# compiled by clang's `-x hip` front end, which is why the sources have to be
# forced back to the CXX language below -- a HIP build enables no CUDA language
# at all, so CMake's extension-based mapping of `.cu` would find no compiler for
# them and fail at generate time.
#
# Keeping device sources in their own STATIC library is what keeps their
# language and `-x hip` flags off the host C++ module compiles, which must not
# see either.
#
# FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY(
#   NAME                  <target, by convention <module>.device>
#   SOURCES               <file.cu ...>   (or SOURCE <file.cu> for one)
#   [LINK_PUBLIC          <lib ...>]
#   [LINK_PRIVATE         <lib ...>]
#   [GPU_OPTIONS          <opt ...>]      compile options, device pass only
#   [INCLUDE_DIRS_PUBLIC  <dir ...>]
#   [INCLUDE_DIRS_PRIVATE <dir ...>]
#   [NO_DEVICE_LINKING]                   separable compilation and
#                                         device-symbol resolution OFF. CUDA
#                                         only -- a HIP build never enables
#                                         either.
# )
macro(FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY)
  set(_bool_opts "NO_DEVICE_LINKING")
  set(_single_opts "NAME;SOURCE")
  set(_multi_opts
      "SOURCES;LINK_PUBLIC;LINK_PRIVATE;GPU_OPTIONS;INCLUDE_DIRS_PUBLIC"
      ";INCLUDE_DIRS_PRIVATE"
  )
  cmake_parse_arguments(
    ARG "${_bool_opts}" "${_single_opts}" "${_multi_opts}" ${ARGN}
  )

  if(NOT ARG_NAME)
    message(
      FATAL_ERROR "FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY: NAME is required"
    )
  endif()

  if(ARG_SOURCES)
    set(_gdl_sources ${ARG_SOURCES})
  elseif(ARG_SOURCE)
    set(_gdl_sources ${ARG_SOURCE})
  else()
    message(
      FATAL_ERROR
        "FREE_BREAKFAST_ADD_GPU_DEVICE_LIBRARY: SOURCE or SOURCES is required"
    )
  endif()

  add_library(${ARG_NAME} STATIC ${_gdl_sources})

  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                       $<INSTALL_INTERFACE:include>
  )

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

  set_target_properties(${ARG_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
    # `.cu` maps to the CUDA language by extension, so there is nothing to set
    # for the language itself.
    target_include_directories(
      ${ARG_NAME} SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
    )
    set_target_properties(
      ${ARG_NAME} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    )
    if(ARG_NO_DEVICE_LINKING)
      set_target_properties(
        ${ARG_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                               CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      )
    else()
      set_target_properties(
        ${ARG_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                               CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
    endif()
    if(ARG_GPU_OPTIONS)
      target_compile_options(
        ${ARG_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARG_GPU_OPTIONS}>
      )
    endif()
  else()
    # A HIP build enables no CUDA language, so `.cu` has to be forced back to
    # CXX. That is also what lets hip::device's $<COMPILE_LANGUAGE:CXX>-gated
    # `-x hip` reach these sources. (A macro opens no directory scope, so these
    # relative names resolve in the caller's directory, which is where
    # set_source_files_properties looks for them.)
    set_source_files_properties(${_gdl_sources} PROPERTIES LANGUAGE CXX)

    # hip::device PRIVATE, never PUBLIC or INTERFACE: its `-x hip` must reach
    # this library's own sources and nothing else. A consumer picking it up
    # transitively would start compiling its plain C++ as HIP.
    target_link_libraries(${ARG_NAME} PRIVATE hip::device)

    # The amdgcn-link step invokes its device linker as the bare name `lld`,
    # which only ROCm's bundled LLVM ships -- the apt clang-20 this project
    # builds with does not. Point the driver at ROCm's own bin directory,
    # derived from where find_package(hip) actually resolved rather than from a
    # hardcoded /opt/rocm.
    get_target_property(
      _gdl_hip_include_dir hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES
    )
    get_filename_component(_gdl_hip_prefix "${_gdl_hip_include_dir}" DIRECTORY)
    target_compile_options(${ARG_NAME} PRIVATE "-B${_gdl_hip_prefix}/llvm/bin")

    # Silence the benign amdgcn "different data layouts" warning, which is a
    # clang-version skew between the apt LLVM and ROCm's bundled one rather
    # than a defect. Drop it once the two converge.
    target_compile_options(${ARG_NAME} PRIVATE -Wno-linker-warnings)

    if(ARG_GPU_OPTIONS)
      target_compile_options(${ARG_NAME} PRIVATE ${ARG_GPU_OPTIONS})
    endif()

    # The CUDA build gets the object cache language-wide, through
    # CMAKE_CUDA_COMPILER_LAUNCHER; under HIP it has to be per target -- see
    # the "Compiler cache" section of the root CMakeLists.txt. It is empty
    # exactly when sccache is absent, and then so is the CXX launcher this
    # replaces. Being CXX, these sources would also be scanned for modules and
    # compiled with an `@<modmap>` argument whose contents the cache cannot
    # see; a `.cu` imports nothing (nvcc cannot), so there is nothing to scan.
    set_target_properties(
      ${ARG_NAME} PROPERTIES CXX_SCAN_FOR_MODULES OFF CXX_COMPILER_LAUNCHER
                             "${FREE_BREAKFAST_GPU_OBJECT_CACHE}"
    )

    unset(_gdl_hip_include_dir)
    unset(_gdl_hip_prefix)
  endif()

  _free_breakfast_create_alias(${ARG_NAME})

  # Joins the free_breakfast_device_libraries umbrella that
  # devtools/cross-backend-check.sh --device-only builds.
  _free_breakfast_join_device_umbrella(${ARG_NAME})

endmacro()
