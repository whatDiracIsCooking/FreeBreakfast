# The ctest core_kernel_stack_bound: no kernel in BINARY needs more stack per
# thread than DeviceResources::kKernelStackBytes, read from SOURCE.
#
# An uncapped memory_budget() leaves room for that much stack on every
# resident thread before the kernel that needs it has run (CUDA reserves it
# outside the pool at its first launch). A regenerated kernel past it would
# take memory the budget had handed to the pool, so this fails naming the
# kernel, and the constant is raised to match. CUDA only: ROCm sizes scratch
# per dispatch, and stack_growth() is 0 there.
#
#   cmake -DCUOBJDUMP=... -DBINARY=... -DSOURCE=... -P kernel_stack_bound.cmake
foreach(_var CUOBJDUMP BINARY SOURCE)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "kernel_stack_bound: ${_var} is not set")
  endif()
endforeach()

file(READ "${SOURCE}" _source)
string(REGEX MATCH "kKernelStackBytes = ([0-9]+);" _match "${_source}")
if(NOT _match)
  message(FATAL_ERROR "kernel_stack_bound: no kKernelStackBytes in ${SOURCE}")
endif()
set(_bound ${CMAKE_MATCH_1})

execute_process(
  COMMAND "${CUOBJDUMP}" --dump-resource-usage "${BINARY}"
  OUTPUT_VARIABLE _usage
  RESULT_VARIABLE _status
  ERROR_VARIABLE _error
)
if(NOT _status EQUAL 0)
  message(FATAL_ERROR "kernel_stack_bound: cuobjdump failed: ${_error}")
endif()

# Each kernel is a "Function <mangled>:" line, then a line with its STACK.
string(REPLACE "\n" ";" _lines "${_usage}")
set(_max 0)
set(_kernels 0)
set(_function "")
set(_worst "")
foreach(_line IN LISTS _lines)
  if(_line MATCHES "Function ([^:]+):")
    set(_function "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "STACK:([0-9]+)")
    math(EXPR _kernels "${_kernels} + 1")
    if(CMAKE_MATCH_1 GREATER _max)
      set(_max ${CMAKE_MATCH_1})
      set(_worst "${_function}")
    endif()
  endif()
endforeach()

if(_kernels EQUAL 0)
  message(FATAL_ERROR "kernel_stack_bound: cuobjdump listed no kernel")
endif()
if(_max GREATER _bound)
  message(
    FATAL_ERROR
      "kernel_stack_bound: ${_worst} needs ${_max} bytes of stack per "
      "thread, past DeviceResources::kKernelStackBytes = ${_bound}: raise it"
  )
endif()
message(
  STATUS
    "kernel_stack_bound: ${_kernels} kernels, the largest stack ${_max} "
    "bytes (${_worst}), within ${_bound}"
)
