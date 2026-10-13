# FREE_BREAKFAST_INTERNAL_HELPERS.cmake Shared internal helpers used by multiple
# free-breakfast macros. Not part of the public API.

# ---------------------------------------------------------------------------
# Internal helpers (underscore-prefixed — not part of the public API)
# ---------------------------------------------------------------------------

# Create a :: alias for a dot-separated target name.
function(_free_breakfast_create_alias target)
  string(FIND "${target}" "." _has_dot)
  if(_has_dot EQUAL -1)
    return()
  endif()
  string(REPLACE "." "::" _alias "${target}")
  add_library(${_alias} ALIAS ${target})
endfunction()

# Add explicit include directories (PUBLIC and/or PRIVATE).
function(_free_breakfast_configure_include_dirs target)
  cmake_parse_arguments(_inc "" "" "PUBLIC;PRIVATE" ${ARGN})
  set(args "")
  if(_inc_PUBLIC)
    list(APPEND args PUBLIC ${_inc_PUBLIC})
  endif()
  if(_inc_PRIVATE)
    list(APPEND args PRIVATE ${_inc_PRIVATE})
  endif()
  target_include_directories(${target} ${args})
endfunction()

# Link PUBLIC and/or PRIVATE dependencies.
function(_free_breakfast_configure_link_libraries target)
  cmake_parse_arguments(_link "" "" "PUBLIC;PRIVATE" ${ARGN})
  set(args "")
  if(_link_PUBLIC)
    list(APPEND args PUBLIC ${_link_PUBLIC})
  endif()
  if(_link_PRIVATE)
    list(APPEND args PRIVATE ${_link_PRIVATE})
  endif()
  target_link_libraries(${target} ${args})
endfunction()

# Add a device library to the free_breakfast_device_libraries umbrella -- the
# target devtools/cross-backend-check.sh --device-only builds
# (CROSS_CHECK_DEVICE_TARGETS in devtools/config.sh), so a new .cu library is
# cross-checked without editing that list. Created by the first call, so a tree
# with no .cu has no umbrella, and --device-only fails there rather than
# building nothing and reporting PASS.
function(_free_breakfast_join_device_umbrella target)
  if(NOT TARGET free_breakfast_device_libraries)
    add_custom_target(
      free_breakfast_device_libraries
      COMMENT "Every free_breakfast_add_gpu_device_library target"
    )
  endif()
  add_dependencies(free_breakfast_device_libraries ${target})
endfunction()
