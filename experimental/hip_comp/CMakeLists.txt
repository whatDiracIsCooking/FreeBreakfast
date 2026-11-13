# GPU lossless compression through hipCOMP: batched LZ4 and Cascaded over
# device arrays, round-tripped bit for bit. HIP only, and only where hipCOMP is
# installed: AMD packages no hipCOMP and this tree's image does not build it,
# so it is found off CMAKE_PREFIX_PATH or skipped (README.md has the recipe).
#
# Raw hipCOMP, not wwr.comp: the portable layer needs WWR_WITH_COMP for the
# whole tree, which the CUDA side's nvCOMP cannot satisfy here.
if(NOT FREE_BREAKFAST_GPU_BACKEND STREQUAL "HIP")
  return()
endif()

find_package(hipcomp CONFIG QUIET)
if(NOT hipcomp_FOUND)
  message(
    STATUS
      "experimental/hip_comp: hipCOMP not found -- demo skipped (see README.md)"
  )
  return()
endif()

# A plain C++ executable, not a module unit: the vendor headers are included,
# and nothing here imports std or a module.
add_executable(free_breakfast_hip_comp_demo hip_comp_demo.cpp)
set_target_properties(
  free_breakfast_hip_comp_demo PROPERTIES CXX_SCAN_FOR_MODULES OFF
)
target_link_libraries(
  free_breakfast_hip_comp_demo PRIVATE hipcomp::hipcomp hip::host
                                       free_breakfast::core::primitives_header
)
