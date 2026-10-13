// GPU cases: these drive a real device, so each one opens with
// FBF_REQUIRE_GPU() and SKIPS rather than fails where there is none.
// Add new device-dependent TUs as `*_gpu.cpp` and list them in CMakeLists.txt's
// gpu executable.
//
// Note these are ordinary .cpp compiled by the C++ front end, not .cu: they
// reach the device through WarpWraps' module surface rather than by being
// device translation units. A `.cu` of actual kernel code belongs in its own
// library, declared with free_breakfast_add_gpu_device_library.
//
// Deliberately narrow. This exercises the runtime-query surface and nothing
// else, because that is enough to prove what a starter needs to prove: the
// submodule linked, the selected backend's runtime loaded, and the skip path
// works on a device-less box. The buffer/stream/BLAS layers, bound to this
// build's error policy, are exercised in gpu_policy_gpu.cpp.

// gtest's headers come FIRST, before any import -- see version_host.cpp for
// why. gpu_requirement.h needs GTEST_SKIP, so it follows gtest.h.
#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

TEST(Device, RuntimeReportsADevice) {
  FBF_REQUIRE_GPU();

  std::int32_t count = 0;
  ASSERT_EQ(wwr::wwrGetDeviceCount(&count), wwr::wwrSuccess);
  EXPECT_GE(count, 1);
}

// Proves the project's own module links alongside the backend runtime in one
// binary -- the combination that breaks first when the module graph or the
// backend switch is wrong.
TEST(Device, ProjectModuleLinksBesideTheRuntime) {
  FBF_REQUIRE_GPU();

  EXPECT_FALSE(free_breakfast::core::version_string().empty());
}

} // namespace
