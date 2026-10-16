// Host-only cases for resources:sys_env: the startup hardware inventory. These need
// no device -- a device-less runtime yields an empty GPU roster, not an error --
// so they run everywhere. The roster's contents on a real device are checked in
// sys_env_gpu.cpp.

#include <gtest/gtest.h>

import std;
import free_breakfast.resources;

namespace {

namespace fb = free_breakfast;

TEST(SysEnv, LogicalCoresIsAtLeastOne) {
  EXPECT_GE(fb::logical_cpu_cores(), 1);
}

TEST(SysEnv, QueryReportsPlausibleHostFigures) {
  auto env = fb::query_sys_env();
  ASSERT_TRUE(env.has_value()) << env.error().message;

  EXPECT_EQ(env->cpu_logical_cores, fb::logical_cpu_cores());
  EXPECT_GE(env->cpu_physical_cores, 1);
  // Hyperthreads share a physical core, never the reverse.
  EXPECT_LE(env->cpu_physical_cores, env->cpu_logical_cores);
  EXPECT_GT(env->host_ram_bytes, 0U);
  EXPECT_GT(env->disk_available_bytes, 0U);
}

// gpu_count is redundant with the roster; it must never drift from it, and the
// roster must agree with the bare count query, device or no device.
TEST(SysEnv, GpuCountMatchesRosterAndRuntime) {
  auto env = fb::query_sys_env();
  ASSERT_TRUE(env.has_value()) << env.error().message;

  EXPECT_EQ(static_cast<std::size_t>(env->gpu_count), env->gpus.size());
  EXPECT_EQ(env->gpu_count, fb::available_gpus());
}

TEST(SysEnv, LogDoesNotThrow) {
  auto env = fb::query_sys_env();
  ASSERT_TRUE(env.has_value()) << env.error().message;
  EXPECT_NO_THROW(fb::log_sys_env(*env));
}

} // namespace
