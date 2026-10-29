/// @file
/// @brief Skip-if-no-device helper shared by the GPU-dependent test files.
///
/// Lives in test/support/ (target free_breakfast::test::support), the one home
/// of the helpers every gtest tree shares; consumers include it by bare name.
///
/// This defines a MACRO, so it is expanded at the point of use and needs the
/// wwr* runtime names visible THERE, not here -- which is why this header
/// imports nothing and why its own include position does not matter. What does
/// matter is that the using TU has `import free_breakfast.core.wwr;` (or
/// `free_breakfast.core`, which re-exports it) in scope at the call site. (Include it next to gtest.h, before the imports, the way every
/// test TU in deps/WarpWraps orders them.)
///
/// Backend-neutral by construction: `wwrGetDeviceCount` is WarpWraps' neutral
/// spelling, bound to cudaGetDeviceCount under a CUDA build and
/// hipGetDeviceCount under a HIP one, so this one macro serves both backends
/// and no test has to know which it is running on.
///
/// Why a skip rather than a failure: ctest runs this binary on whatever
/// machine built it, and the build host is routinely not the run host. A GPU
/// test that FAILS on a GPU-less box makes the whole suite unusable there; one
/// that skips keeps the host-side coverage meaningful. The cost is that a green
/// run on a laptop proves much less than a green run on a GPU box -- which is
/// why devtools/cpp-tier.sh records the ctest output in full, and why `-rs` is
/// the habit on the Python side.
///
/// Note the error check is not folded away: a runtime that cannot be queried
/// at all (no driver, a library mismatch) returns non-success and leaves the
/// count untouched, so testing only the count would read uninitialised memory.

#pragma once

#include <gtest/gtest.h>

#define FBF_REQUIRE_GPU()                                                                          \
  do {                                                                                             \
    std::int32_t fb_device_count = 0;                                                              \
    if (::wwr::wwrGetDeviceCount(&fb_device_count) != ::wwr::wwrSuccess || fb_device_count < 1) {  \
      GTEST_SKIP() << "no GPU device visible to the runtime";                                      \
    }                                                                                              \
  } while (0)
