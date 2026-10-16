// GPU cases for NodeResources: one DeviceResources per visible device, the
// caller's current device left alone by construction, and on_device()
// selecting the device on whichever executor worker runs the task. Skips where
// there is no device; the host half is covered in node_resources_host.cpp.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

std::int32_t device_count() {
  std::int32_t count = 0;
  EXPECT_EQ(wwr::wwrGetDeviceCount(&count), wwr::wwrSuccess);
  return count;
}

std::int32_t current_device() {
  std::int32_t idx = -1;
  EXPECT_EQ(wwr::wwrGetDevice(&idx), wwr::wwrSuccess);
  return idx;
}

// A small worker count is enough here and keeps the cases light on a big box.
constexpr std::int32_t kWorkers = 2;

TEST(NodeResourcesGpu, DefaultTakesEveryVisibleDeviceInOrder) {
  FBF_REQUIRE_GPU();

  const fb::NodeResources node{{.num_threads = kWorkers}};

  ASSERT_EQ(node.device_count(), static_cast<std::size_t>(device_count()));
  for (std::size_t i = 0; i < node.device_count(); ++i) {
    const auto &res = node.device(i);
    EXPECT_EQ(res.dev_idx(), static_cast<std::int32_t>(i));
    EXPECT_FALSE(std::string_view(res.device()->props().name).empty()) << "device " << i;
    EXPECT_GT(res.device()->props().totalGlobalMem, 0U) << "device " << i;
  }
}

TEST(NodeResourcesGpu, ExplicitDeviceListAndThreshold) {
  FBF_REQUIRE_GPU();

  constexpr std::uint64_t kThreshold = std::uint64_t{64} << 20;
  const fb::NodeResources node{{.devices = std::vector<std::int32_t>{0},
                                .num_threads = kWorkers,
                                .pool_release_threshold = kThreshold}};

  ASSERT_EQ(node.device_count(), 1U);
  EXPECT_EQ(node.device(0).dev_idx(), 0);

  std::uint64_t threshold = 0;
  ASSERT_EQ(wwr::wwrMemPoolGetAttribute(node.device(0).device()->pool(),
                                        wwr::wwrMemPoolAttrReleaseThreshold, &threshold),
            wwr::wwrSuccess);
  EXPECT_EQ(threshold, kThreshold);
}

// Creating a device's handles leaves that device current; NodeResources must
// put the caller's back. With one device that is the same index, so the case
// that can tell the difference is the two-device one.
TEST(NodeResourcesGpu, ConstructionLeavesTheCurrentDeviceAlone) {
  FBF_REQUIRE_GPU();
  const std::int32_t count = device_count();
  const std::int32_t start = count - 1;

  ASSERT_EQ(wwr::wwrSetDevice(start), wwr::wwrSuccess);
  {
    const fb::NodeResources node{{.num_threads = kWorkers}};
    EXPECT_EQ(current_device(), start);
  }
  EXPECT_EQ(current_device(), start);
  ASSERT_EQ(wwr::wwrSetDevice(0), wwr::wwrSuccess);
}

TEST(NodeResourcesGpu, OnDeviceSelectsTheDeviceOnAWorker) {
  FBF_REQUIRE_GPU();

  fb::NodeResources node{{.num_threads = kWorkers}};

  for (std::size_t i = 0; i < node.device_count(); ++i) {
    auto seen = node.executor()
                    .async([&node, i] {
                      const std::int32_t inside =
                          node.on_device(i, [](fb::DeviceResources &) { return current_device(); });
                      return std::pair{inside, node.device(i).dev_idx()};
                    })
                    .get();
    EXPECT_EQ(seen.first, seen.second) << "device position " << i;
  }
}

// The point of the executor: real device work fanned out from host tasks. Each
// task round-trips a buffer through its device on that device's stream.
TEST(NodeResourcesGpu, TasksDriveEveryDevice) {
  FBF_REQUIRE_GPU();

  fb::NodeResources node{{.num_threads = kWorkers}};
  constexpr std::size_t kCount = 1024;

  std::vector<std::vector<float>> results(node.device_count());
  fb::tf::Taskflow flow;
  for (std::size_t i = 0; i < node.device_count(); ++i) {
    flow.emplace([&node, &results, i] {
      node.on_device(i, [&](fb::DeviceResources &res) {
        fb::Buffers::host<float> in(kCount);
        fb::Buffers::host<float> out(kCount);
        for (std::size_t k = 0; k < kCount; ++k)
          in.data()[k] = static_cast<float>(k + i);
        fb::Buffers::device<float> d(kCount, res.device());

        // Qualified: a bare wwrStream_t carries no ADL into wwr::extension.
        const wwr::wwrStream_t stream = res.stream();
        EXPECT_EQ(ext::copy(d, in, stream), wwr::wwrSuccess);
        EXPECT_EQ(ext::copy(out, d, stream), wwr::wwrSuccess);
        EXPECT_EQ(ext::sync(stream), wwr::wwrSuccess);
        results[i].assign(out.data(), out.data() + kCount);
      });
    });
  }
  node.executor().run(flow).wait();

  for (std::size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].size(), kCount) << "device position " << i;
    for (std::size_t k = 0; k < kCount; ++k)
      ASSERT_EQ(results[i][k], static_cast<float>(k + i)) << "device " << i << " at " << k;
  }
}

TEST(NodeResourcesGpu, LogListsDevicesWithoutThrowing) {
  FBF_REQUIRE_GPU();

  const fb::NodeResources node{{.num_threads = kWorkers}};
  EXPECT_NO_THROW(node.log());
}

} // namespace
