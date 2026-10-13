# FREE_BREAKFAST_ADD_INTERFACE_LIBRARY.cmake Provides
# FREE_BREAKFAST_ADD_INTERFACE_LIBRARY macro for creating INTERFACE libraries
# that expose header files (.h/.cuh) to consumers

# ---------------------------------------------------------------------------
# Public macro
# ---------------------------------------------------------------------------

# Macro to create an INTERFACE library with standard boilerplate. Usage:
# FREE_BREAKFAST_ADD_INTERFACE_LIBRARY( NAME library_name )
#
# Parameters: NAME - Target name (e.g. free_breakfast.core.parallel_for)
macro(FREE_BREAKFAST_ADD_INTERFACE_LIBRARY)
  set(_single_opts "NAME")
  set(_multi_opts "LINK_PUBLIC")
  cmake_parse_arguments(ARG "" "${_single_opts}" "${_multi_opts}" ${ARGN})

  if(NOT ARG_NAME)
    message(
      FATAL_ERROR "FREE_BREAKFAST_ADD_INTERFACE_LIBRARY: NAME is required"
    )
  endif()

  add_library(${ARG_NAME} INTERFACE)

  target_include_directories(
    ${ARG_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                          $<INSTALL_INTERFACE:include>
  )

  if(ARG_LINK_PUBLIC)
    target_link_libraries(${ARG_NAME} INTERFACE ${ARG_LINK_PUBLIC})
  endif()

  _free_breakfast_create_alias(${ARG_NAME})

endmacro()
