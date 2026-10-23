# The compiled-in basis sets: every data/basis/*.json, wrapped verbatim in a
# raw string literal inside a generated module implementation unit. See
# builtin_basis.cpp.in for why the data is compiled in rather than installed.
#
# Generated at CONFIGURE time, not build time, so the file exists before the
# target below names it as a source. CONFIGURE_DEPENDS re-runs the glob on
# every build, so an added or removed .json is noticed; the
# CMAKE_CONFIGURE_DEPENDS property below re-runs configure when the contents
# of one change.
set(_fb_basis_dir "${PROJECT_SOURCE_DIR}/data/basis")
file(GLOB _fb_basis_files CONFIGURE_DEPENDS "${_fb_basis_dir}/*.json")
if(NOT _fb_basis_files)
  message(FATAL_ERROR "no basis sets found in ${_fb_basis_dir}")
endif()

# A document holding the raw literal's closing sequence would end the literal
# early and spray JSON into the translation unit, so it is checked rather than
# assumed. Split in two so this file does not contain the sequence either.
set(_fb_raw_close ")fbbasis" "\"")
string(JOIN "" _fb_raw_close ${_fb_raw_close})

set(_fb_basis_entries "")
list(LENGTH _fb_basis_files _fb_basis_count)
foreach(_fb_basis_file IN LISTS _fb_basis_files)
  file(READ "${_fb_basis_file}" _fb_basis_json)
  string(FIND "${_fb_basis_json}" "${_fb_raw_close}" _fb_raw_close_at)
  if(NOT _fb_raw_close_at EQUAL -1)
    message(
      FATAL_ERROR
        "${_fb_basis_file} contains the raw-literal terminator; "
        "change the delimiter in src/io/basis/CMakeLists.txt"
    )
  endif()
  # The name the document calls itself, so the table's key cannot drift from
  # the data. Doubles as a JSON syntax check on the committed file.
  string(JSON _fb_basis_name GET "${_fb_basis_json}" "name")
  string(APPEND _fb_basis_entries "    {\"${_fb_basis_name}\", "
         "R\"fbbasis(${_fb_basis_json})fbbasis\"},\n"
  )
endforeach()

# string(REPLACE) rather than configure_file(), so nothing scans the basis data
# itself for substitutions: the JSON is only ever a replacement value.
# configure_file(COPYONLY) then rewrites the real output only when it differs,
# which keeps an unchanged reconfigure from recompiling 500 KB of literals.
set(_fb_builtin_basis "${CMAKE_CURRENT_BINARY_DIR}/builtin_basis.cpp")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/builtin_basis.cpp.in" _fb_basis_template)
string(REPLACE "@FB_BUILTIN_BASIS_COUNT@" "${_fb_basis_count}"
               _fb_basis_template "${_fb_basis_template}"
)
string(REPLACE "@FB_BUILTIN_BASIS_ENTRIES@" "${_fb_basis_entries}"
               _fb_basis_template "${_fb_basis_template}"
)
file(WRITE "${_fb_builtin_basis}.in-progress" "${_fb_basis_template}")
configure_file("${_fb_builtin_basis}.in-progress" "${_fb_builtin_basis}"
               COPYONLY
)

set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb_basis_files}
           "${CMAKE_CURRENT_SOURCE_DIR}/builtin_basis.cpp.in"
)

# basis and molecule are PUBLIC: the reader's signatures name BasisStamp,
# BasisSet and Molecule.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.io.basis
  PRIMARY_INTERFACE
  basis.cppm
  PARTITIONS
  basis_reader.cppm
  IMPLEMENTATION
  basis_reader.cpp
  "${_fb_builtin_basis}"
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  free_breakfast::molecule
  LINK_PRIVATE
  free_breakfast::io::base
  free_breakfast::io::json
  free_breakfast::core::diagnostics_macros
  IMPORT_STD
)
