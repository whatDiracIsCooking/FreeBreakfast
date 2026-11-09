# Run one examples/ config through the free-breakfast binary, as a user would,
# and check what it left behind. A ctest entry of test/examples/CMakeLists.txt:
#
#   cmake -DBINARY=<free-breakfast> -DINPUT_DIR=<examples/> -DCONFIG=<x.toml>
#         -DOUTPUT_DIR=<dir> -DFILES=<a,b,...> [-DJSON=<file.json>
#         -DKEYS=<k1,k2,...>] [-DEXPECT=<regex>] -P run_example.cmake
#
# Fails when the binary exits non-zero, when it does not print its closing
# `completed: 0 errors` line, when one of FILES is missing from OUTPUT_DIR,
# when JSON does not parse or lacks one of KEYS at its top level, or when its
# output does not match EXPECT. A config that
# stops parsing fails here on any machine: main reads and plans it before it
# looks for a device.
#
# With no device, main logs `Devices: 0` once the config is planned, and the
# run cannot go further. That is reported as the line SKIP_REGULAR_EXPRESSION
# matches, so the entry skips rather than fails, as FBF_REQUIRE_GPU() does for a
# gtest case.
foreach(_var BINARY INPUT_DIR CONFIG OUTPUT_DIR FILES)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "run_example.cmake needs -D${_var}=")
  endif()
endforeach()

# A stale file from an earlier run must not pass for this one's.
file(REMOVE_RECURSE "${OUTPUT_DIR}")
execute_process(
  COMMAND "${BINARY}" -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" "${CONFIG}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
message("${_out}")

if(_out MATCHES "Devices: +0[\r\n]")
  message("run_example: no GPU device visible to the runtime")
  return()
endif()
if(NOT _rc STREQUAL "0")
  message(FATAL_ERROR "free-breakfast ${CONFIG} exited ${_rc}")
endif()
if(NOT _out MATCHES "completed: 0 errors")
  message(FATAL_ERROR "free-breakfast ${CONFIG}: no `completed: 0 errors`")
endif()

if(DEFINED EXPECT AND NOT _out MATCHES "${EXPECT}")
  message(FATAL_ERROR "free-breakfast ${CONFIG}: no line matching `${EXPECT}`")
endif()

string(REPLACE "," ";" _files "${FILES}")
foreach(_file IN LISTS _files)
  if(NOT EXISTS "${OUTPUT_DIR}/${_file}")
    message(FATAL_ERROR "free-breakfast ${CONFIG}: no ${OUTPUT_DIR}/${_file}")
  endif()
endforeach()

if(DEFINED JSON)
  file(READ "${OUTPUT_DIR}/${JSON}" _doc)
  string(REPLACE "," ";" _keys "${KEYS}")
  foreach(_key IN LISTS _keys)
    string(JSON _type ERROR_VARIABLE _error TYPE "${_doc}" "${_key}")
    if(_error)
      message(FATAL_ERROR "${JSON}: no key '${_key}' (${_error})")
    endif()
  endforeach()
endif()
