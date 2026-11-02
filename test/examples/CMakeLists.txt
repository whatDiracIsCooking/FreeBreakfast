# ---------------------------------------------------------------------------
# The examples/ configs, run through the free-breakfast binary
# ---------------------------------------------------------------------------
# Every other pipeline test drives the planner through its API; these are the
# one path that goes through main.cpp's argument handling, its file I/O and
# its printout, on configs a user is pointed at (examples/README.md). Each
# entry is run_example.cmake over one config: the exit status, the closing
# `completed: 0 errors` line, the files the run writes, and a JSON's keys.
#
# Two of examples/, the cheapest that between them write a Molden file and a
# JSON one; the rest are run by hand (examples/README.md). Each is about a
# second or two on the card, inside test/support/slow_test.h's rule.
#
# The entries are GPU ones, so they carry the `gpu` label the ci-* test
# presets exclude, and skip rather than fail with no device (run_example.cmake
# says how). Their command is cmake, not a GPU test binary, so
# devtools/split-gpu-entries.py leaves them to ctest, beside the host entries.
set(_fb_examples "${PROJECT_SOURCE_DIR}/examples")

# example_<name> runs examples/<name>.toml into the build tree's <name>/.
set(_fb_driver -P ${CMAKE_CURRENT_SOURCE_DIR}/run_example.cmake)
add_test(
  NAME example_rhf
  COMMAND
    ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:free_breakfast_main>
    -DINPUT_DIR=${_fb_examples} -DCONFIG=${_fb_examples}/rhf.toml
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/rhf -DFILES=scf.molden
    ${_fb_driver}
)
add_test(
  NAME example_hessian
  COMMAND
    ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:free_breakfast_main>
    -DINPUT_DIR=${_fb_examples} -DCONFIG=${_fb_examples}/hessian.toml
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/hessian
    -DFILES=hessian.json -DJSON=hessian.json
    -DKEYS=method,hessian,frequencies,modes,dipole_derivative,intensities
    ${_fb_driver}
)
set_tests_properties(
  example_rhf example_hessian
  PROPERTIES LABELS gpu TIMEOUT ${_fb_test_timeout} SKIP_REGULAR_EXPRESSION
             "run_example: no GPU device visible to the runtime"
)
