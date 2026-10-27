# The compiled-in basis sets: every data/basis/*.json (the orbital sets) and
# every data/aux_basis/*.json (the density-fitting sets), each wrapped verbatim
# in a raw string literal inside a generated module implementation unit. See
# builtin_basis.cpp.in for why the data is compiled in rather than installed,
# and why the fitting sets get a table of their own.
#
# Generated at CONFIGURE time, not build time, so the file exists before the
# target below names it as a source. CONFIGURE_DEPENDS re-runs the glob on
# every build, so an added or removed .json is noticed; the
# CMAKE_CONFIGURE_DEPENDS property below re-runs configure when the contents
# of one change.

# A document holding the raw literal's closing sequence would end the literal
# early and spray JSON into the translation unit, so it is checked rather than
# assumed. Split in two so this file does not contain the sequence either.
set(_fb_raw_close ")fbbasis" "\"")
string(JOIN "" _fb_raw_close ${_fb_raw_close})

set(_fb_basis_template_in "${CMAKE_CURRENT_SOURCE_DIR}/builtin_basis.cpp.in")
file(READ "${_fb_basis_template_in}" _fb_basis_template)
set(_fb_basis_all_files "")

# One table per data directory: the template's @FB_<token>_COUNT@ and
# @FB_<token>_ENTRIES@ become its size and its {name, json} entries.
foreach(_fb_table IN ITEMS "basis:BUILTIN_BASIS" "aux_basis:BUILTIN_AUX_BASIS")
  string(REPLACE ":" ";" _fb_table "${_fb_table}")
  list(GET _fb_table 0 _fb_basis_subdir)
  list(GET _fb_table 1 _fb_basis_token)
  set(_fb_basis_dir "${PROJECT_SOURCE_DIR}/data/${_fb_basis_subdir}")
  file(GLOB _fb_basis_files CONFIGURE_DEPENDS "${_fb_basis_dir}/*.json")
  if(NOT _fb_basis_files)
    message(FATAL_ERROR "no basis sets found in ${_fb_basis_dir}")
  endif()
  list(APPEND _fb_basis_all_files ${_fb_basis_files})

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
    # The name the document calls itself, so the table's key cannot drift
    # from the data. Doubles as a JSON syntax check on the committed file.
    string(JSON _fb_basis_name GET "${_fb_basis_json}" "name")
    string(APPEND _fb_basis_entries "    {\"${_fb_basis_name}\", "
           "R\"fbbasis(${_fb_basis_json})fbbasis\"},\n"
    )
  endforeach()

  # string(REPLACE) rather than configure_file(), so nothing scans the basis
  # data itself for substitutions: the JSON is only ever a replacement value.
  string(REPLACE "@FB_${_fb_basis_token}_COUNT@" "${_fb_basis_count}"
                 _fb_basis_template "${_fb_basis_template}"
  )
  string(REPLACE "@FB_${_fb_basis_token}_ENTRIES@" "${_fb_basis_entries}"
                 _fb_basis_template "${_fb_basis_template}"
  )
endforeach()

# configure_file(COPYONLY) rewrites the real output only when it differs,
# which keeps an unchanged reconfigure from recompiling 1 MB of literals.
set(_fb_builtin_basis "${CMAKE_CURRENT_BINARY_DIR}/builtin_basis.cpp")
file(WRITE "${_fb_builtin_basis}.in-progress" "${_fb_basis_template}")
configure_file("${_fb_builtin_basis}.in-progress" "${_fb_builtin_basis}"
               COPYONLY
)

set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb_basis_all_files}
           "${_fb_basis_template_in}"
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
