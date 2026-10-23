# The compiled-in SAD stamps: every data/sad/<basis>/<Z>.npy, its bytes escaped
# into a string literal inside a generated module implementation unit. The
# same arrangement as src/io/basis, and generated at configure time for the
# same reasons; see builtin_sad.cpp.in.
#
# An empty data/sad/ is allowed, not fatal: it means a binary that computes
# every atomic density, which is slower but right, and it is the state the
# generator (tools/sad) first runs from.
set(_fb_sad_dir "${PROJECT_SOURCE_DIR}/data/sad")
file(GLOB _fb_sad_files CONFIGURE_DEPENDS "${_fb_sad_dir}/*/*.npy")

# A stamp is looked up by the basis set's canonical name, the BSE "name" of its
# data/basis document. A directory naming no such document could never be
# found, so it is refused here rather than baked in as dead weight.
file(GLOB _fb_sad_basis_files CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/data/basis/*.json"
)
set(_fb_sad_basis_names "")
foreach(_fb_basis_file IN LISTS _fb_sad_basis_files)
  file(READ "${_fb_basis_file}" _fb_basis_json)
  string(JSON _fb_basis_name GET "${_fb_basis_json}" "name")
  list(APPEND _fb_sad_basis_names "${_fb_basis_name}")
endforeach()

set(_fb_sad_entries "")
list(LENGTH _fb_sad_files _fb_sad_count)
foreach(_fb_sad_file IN LISTS _fb_sad_files)
  get_filename_component(_fb_sad_basis_dir "${_fb_sad_file}" DIRECTORY)
  get_filename_component(_fb_sad_basis "${_fb_sad_basis_dir}" NAME)
  get_filename_component(_fb_sad_stem "${_fb_sad_file}" NAME_WE)
  if(NOT _fb_sad_basis IN_LIST _fb_sad_basis_names)
    message(
      FATAL_ERROR
        "${_fb_sad_file}: ${_fb_sad_basis} is not the name of any "
        "data/basis set; regenerate data/sad with tools/sad"
    )
  endif()
  if(NOT _fb_sad_stem MATCHES "^[0-9][0-9][0-9]$")
    message(FATAL_ERROR "${_fb_sad_file}: expected a three-digit Z")
  endif()
  string(REGEX REPLACE "^0+([0-9])" "\\1" _fb_sad_z "${_fb_sad_stem}")
  file(READ "${_fb_sad_file}" _fb_sad_hex HEX)
  string(REGEX REPLACE "(..)" "\\\\x\\1" _fb_sad_bytes "${_fb_sad_hex}")
  string(APPEND _fb_sad_entries "    {\"${_fb_sad_basis}\", ${_fb_sad_z}, "
         "\"${_fb_sad_bytes}\"sv},\n"
  )
endforeach()

# string(REPLACE) then configure_file(COPYONLY), as src/io/basis does: nothing
# scans the data for substitutions, and an unchanged reconfigure leaves the
# generated file's timestamp alone.
set(_fb_builtin_sad "${CMAKE_CURRENT_BINARY_DIR}/builtin_sad.cpp")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/builtin_sad.cpp.in" _fb_sad_template)
string(REPLACE "@FB_BUILTIN_SAD_COUNT@" "${_fb_sad_count}" _fb_sad_template
               "${_fb_sad_template}"
)
string(REPLACE "@FB_BUILTIN_SAD_ENTRIES@" "${_fb_sad_entries}"
               _fb_sad_template "${_fb_sad_template}"
)
file(WRITE "${_fb_builtin_sad}.in-progress" "${_fb_sad_template}")
configure_file("${_fb_builtin_sad}.in-progress" "${_fb_builtin_sad}" COPYONLY)

set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb_sad_files} ${_fb_sad_basis_files}
           "${CMAKE_CURRENT_SOURCE_DIR}/builtin_sad.cpp.in"
)

# Nothing but core::types is in the interface; npy and error_handling are the
# lookup's parse and its abort.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.io.sad
  PRIMARY_INTERFACE
  sad.cppm
  IMPLEMENTATION
  "${_fb_builtin_sad}"
  LINK_PUBLIC
  free_breakfast::core::types
  LINK_PRIVATE
  free_breakfast::core::error_handling
  free_breakfast::io::npy
  IMPORT_STD
)
