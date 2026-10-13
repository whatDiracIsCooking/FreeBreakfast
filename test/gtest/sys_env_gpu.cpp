// GPU cases for core/sys_env: on a box with a device, the GPU roster reports
// every device the runtime counts, with real properties. Skips where there is
// no device; the device-less path is covered in sys_env_host.cpp.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

namespace fb = free_breakfast;

TEST(SysEnvGpu, RosterListsEveryDeviceWithItsProperties) {
  FREE_BREAKFAST_REQUIRE_GPU();

  std::int32_t count = 0;
  ASSERT_EQ(wwr::wwrGetDeviceCount(&count), wwr::wwrSuccess);

  auto env = fb::query_sys_env();
  ASSERT_TRUE(env.has_value()) << env.error().message;
  ASSERT_EQ(env->gpu_count, count);
  ASSERT_EQ(env->gpus.size(), static_cast<std::size_t>(count));

  for (std::size_t i = 0; i < env->gpus.size(); ++i) {
    const auto &gpu = env->gpus[i];
    EXPECT_EQ(gpu.index, static_cast<std::int32_t>(i));
    EXPECT_FALSE(gpu.name.empty()) << "device " << i;
    EXPECT_GT(gpu.total_memory_bytes, 0U) << "device " << i;
    EXPECT_GT(gpu.compute_capability_major, 0) << "device " << i;
  }
}

} // namespace
