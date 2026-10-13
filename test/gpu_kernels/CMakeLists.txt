# ---------------------------------------------------------------------------
# Device kernels the GPU tests drive
# ---------------------------------------------------------------------------
# Kept apart from test/gtest because a .cu never goes into a test executable
# (those are module-scanned and built with CXX_MODULE_STD, which nvcc
# understands neither of), and because this directory needs no GoogleTest: the
# root CMakeLists adds it even under FREE_BREAKFAST_COMPILE_TIME_ONLY, so
# devtools/cross-backend-check.sh compiles these kernels for the backend this
# build is not targeting.
#
# primitives_kernels.cu includes primitives.h, which links wwr::device (the
# vendor runtime header and wwr's runtime.h, whose device section supplies the
# wwr* allocation / copy / launch names) and types_headers.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.test.gpu_kernels
  SOURCES
  primitives_kernels.cu
  LINK_PUBLIC
  free_breakfast::core::types_headers
  LINK_PRIVATE
  free_breakfast::core::primitives_header
)
