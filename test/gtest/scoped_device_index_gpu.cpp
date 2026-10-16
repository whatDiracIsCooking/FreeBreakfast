// GPU cases for core.wwr's ScopedDeviceIndex: the guard makes its device
// current and puts the previous one back. The switch between two distinct
// devices needs a box with two; on a one-device box that case skips.

// gtest's headers come FIRST, before any import -- see version_host.cpp for
// why. gpu_requirement.h needs GTEST_SKIP, so it follows gtest.h.
#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace fb = free_breakfast;

namespace {

std::int32_t current_device() {
  std::int32_t idx = -1;
  EXPECT_EQ(wwr::wwrGetDevice(&idx), wwr::wwrSuccess);
  return idx;
}

TEST(ScopedDeviceIndex, SelectsAndRestoresOnOneDevice) {
  FBF_REQUIRE_GPU();

  ASSERT_EQ(wwr::wwrSetDevice(0), wwr::wwrSuccess);
  {
    const fb::ScopedDeviceIndex on{0};
    EXPECT_EQ(current_device(), 0);
    EXPECT_EQ(on.original_idx, 0);
  }
  EXPECT_EQ(current_device(), 0);
}

TEST(ScopedDeviceIndex, SwitchesBetweenDevicesAndBack) {
  FBF_REQUIRE_GPU();
  std::int32_t count = 0;
  ASSERT_EQ(wwr::wwrGetDeviceCount(&count), wwr::wwrSuccess);
  if (count < 2) {
    GTEST_SKIP() << "needs two GPU devices, found " << count;
  }

  ASSERT_EQ(wwr::wwrSetDevice(0), wwr::wwrSuccess);
  {
    const fb::ScopedDeviceIndex on{1};
    EXPECT_EQ(current_device(), 1);
    {
      const fb::ScopedDeviceIndex inner{0};
      EXPECT_EQ(current_device(), 0);
    }
    EXPECT_EQ(current_device(), 1);
  }
  EXPECT_EQ(current_device(), 0);
}

// The guard is per thread: a switch on a worker leaves the caller's device alone.
TEST(ScopedDeviceIndex, IsPerThread) {
  FBF_REQUIRE_GPU();

  ASSERT_EQ(wwr::wwrSetDevice(0), wwr::wwrSuccess);
  std::int32_t seen_on_worker = -1;
  std::thread worker([&] {
    const fb::ScopedDeviceIndex on{0};
    seen_on_worker = current_device();
  });
  worker.join();
  EXPECT_EQ(seen_on_worker, 0);
  EXPECT_EQ(current_device(), 0);
}

} // namespace
