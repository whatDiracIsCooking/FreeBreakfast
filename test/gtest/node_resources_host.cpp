// Host-only cases for NodeResources: the host inventory it reads and the
// executor it sizes. Every case names an empty device list, so none touches a
// device and they run the same on any box. Device-holding cases are in
// node_resources_gpu.cpp.

#include <gtest/gtest.h>

import std;
import free_breakfast.resources;

namespace {

namespace fb = free_breakfast;

// No devices: the host half of NodeResources on its own.
fb::NodeResourcesConfig host_only(std::optional<std::int32_t> num_threads = std::nullopt) {
  return {.devices = std::vector<std::int32_t>{}, .num_threads = num_threads};
}

static_assert(!std::is_copy_constructible_v<fb::NodeResources>);
static_assert(!std::is_move_constructible_v<fb::NodeResources>);

TEST(NodeResources, ReportsPlausibleHostFigures) {
  const fb::NodeResources node{host_only()};

  EXPECT_EQ(node.logical_cores(),
            std::max(1, static_cast<std::int32_t>(std::thread::hardware_concurrency())));
  EXPECT_GE(node.physical_cores(), 1);
  // Hyperthreads share a physical core, never the reverse.
  EXPECT_LE(node.physical_cores(), node.logical_cores());
  EXPECT_GT(node.host_ram_bytes(), 0U);
}

TEST(NodeResources, DefaultWorkersArePhysicalCoresLessTheReserve) {
  const fb::NodeResources node{host_only()};

  EXPECT_EQ(fb::NodeResources::kReservedCores, 2);
  EXPECT_EQ(node.num_threads(),
            std::max(1, node.physical_cores() - fb::NodeResources::kReservedCores));
}

TEST(NodeResources, ConfiguredWorkerCountIsTaken) {
  fb::NodeResources node{host_only(3)};

  EXPECT_EQ(node.num_threads(), 3);
  EXPECT_EQ(node.executor().num_workers(), 3U);
}

// The message goes to the log's stdout sink, which a death test cannot match.
TEST(NodeResourcesDeathTest, WorkerCountBelowOneAborts) {
  EXPECT_DEATH(fb::NodeResources{host_only(0)}, "");
}

TEST(NodeResources, EmptyDeviceListHoldsNoDevices) {
  const fb::NodeResources node{host_only()};

  EXPECT_EQ(node.device_count(), 0U);
  EXPECT_TRUE(node.devices().empty());
}

TEST(NodeResources, ExecutorRunsATaskGraph) {
  fb::NodeResources node{host_only(4)};

  constexpr std::int32_t kTasks = 64;
  std::atomic<std::int32_t> ran{0};
  fb::tf::Taskflow flow;
  for (std::int32_t i = 0; i < kTasks; ++i) {
    flow.emplace([&ran] { ran.fetch_add(1, std::memory_order_relaxed); });
  }
  node.executor().run(flow).wait();

  EXPECT_EQ(ran.load(), kTasks);
}

TEST(NodeResources, LogDoesNotThrow) {
  const fb::NodeResources node{host_only()};
  EXPECT_NO_THROW(node.log());
}

} // namespace
