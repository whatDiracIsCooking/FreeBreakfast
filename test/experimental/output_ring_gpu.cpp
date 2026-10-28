// GPU cases for experimental.output_ring: do device arrays streamed through the
// ring land on disk byte for byte as io.npy's own writer would put them?
//
// The slots are made tiny (a few dozen doubles) so every array is many chunks,
// and several files are queued before one wait(), so chunks of different files
// share the slots and land out of order. Each file is compared whole against
// npy_to_bytes of the same values: a chunk at the wrong offset, a stale slot or
// a short write is a byte mismatch, not a tolerance.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.core.nontransferrable;
import free_breakfast.io.base;
import free_breakfast.io.npy;
import free_breakfast.experimental.output_ring;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr fb::OutputRingConfig kTinySlots{.num_slots = 3, .slot_bytes = 37 * sizeof(double)};

// A host array of @p shape with distinct values, and its device copy.
struct Array {
  fb::NpyArray host;
  fb::Buffers::device<double> device;
};

Array upload(fb::DeviceResources &res, std::vector<std::size_t> shape, const double seed) {
  const std::size_t count = fb::element_count(shape).value();
  fb::NpyArray host{.shape = std::move(shape), .data = std::vector<double>(count)};
  for (std::size_t i = 0; i < count; ++i) {
    host.data[i] = seed + 0.25 * static_cast<double>(i);
  }
  fb::Buffers::device<double> device{std::max<std::size_t>(count, 1), res.device()};
  EXPECT_EQ(wwr::wwrMemcpyAsync(device.data(), host.data.data(), count * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  return {.host = std::move(host), .device = std::move(device)};
}

std::span<const double> span_of(const Array &a) {
  return {a.device.data(), a.host.data.size()};
}

void expect_file_is(const std::filesystem::path &path, const fb::NpyArray &expected) {
  const auto bytes = fb::read_file(path);
  ASSERT_TRUE(bytes.has_value()) << path;
  const auto want = fb::npy_to_bytes(expected);
  ASSERT_TRUE(want.has_value());
  EXPECT_EQ(*bytes, *want) << path;
}

TEST(OutputRing, StreamsSeveralFilesThroughSharedSlots) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::tf::Executor executor{4};
  const fb::test::ScratchDir dir("output_ring");

  // Shapes below, at and well past one slot (37), in one to three axes.
  const std::vector<std::vector<std::size_t>> shapes{{37, 29}, {200}, {5, 4, 3}, {36}, {37}, {1}};
  std::vector<Array> arrays;
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    arrays.push_back(upload(res, shapes[i], 1000.0 * static_cast<double>(i)));
  }

  fb::OutputRing ring{res.dev_idx(), executor, kTinySlots};
  ASSERT_EQ(ring.slot_elements(), 37U);
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    ASSERT_TRUE(ring.write_npy(res, dir / std::format("{}.npy", i), arrays[i].host.shape,
                               span_of(arrays[i])));
  }
  ASSERT_TRUE(ring.wait());
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    expect_file_is(dir / std::format("{}.npy", i), arrays[i].host);
  }
}

// Driven from inside the executor, as a pipeline step is, with a single
// worker: taking a slot must run the writer tasks rather than block on them.
TEST(OutputRing, ProgressesFromInsideASingleWorkerExecutor) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::tf::Executor executor{1};
  const fb::test::ScratchDir dir("output_ring");
  const Array a = upload(res, {50, 40}, 0.5);
  const Array b = upload(res, {333}, -7.0);

  fb::OutputRing ring{res.dev_idx(), executor, kTinySlots};
  fb::Status written;
  fb::tf::Taskflow flow;
  flow.emplace([&] {
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    written =
        ring.write_npy(res, dir / "a.npy", a.host.shape, span_of(a))
            .and_then([&] { return ring.write_npy(res, dir / "b.npy", b.host.shape, span_of(b)); })
            .and_then([&] { return ring.wait(); });
  });
  executor.run(flow).wait();
  ASSERT_TRUE(written) << written.error().message;
  expect_file_is(dir / "a.npy", a.host);
  expect_file_is(dir / "b.npy", b.host);
}

TEST(OutputRing, WritesAnEmptyArrayAsItsHeader) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::tf::Executor executor{2};
  const fb::test::ScratchDir dir("output_ring");
  const Array a = upload(res, {0, 3}, 0.0);

  fb::OutputRing ring{res.dev_idx(), executor, kTinySlots};
  ASSERT_TRUE(ring.write_npy(res, dir / "empty.npy", a.host.shape, span_of(a)));
  ASSERT_TRUE(ring.wait());
  expect_file_is(dir / "empty.npy", a.host);
}

TEST(OutputRing, RejectsAShapeThatDisagreesWithTheBuffer) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::tf::Executor executor{2};
  const fb::test::ScratchDir dir("output_ring");
  const Array a = upload(res, {6, 6}, 0.0);

  fb::OutputRing ring{res.dev_idx(), executor, kTinySlots};
  const std::vector<std::size_t> wrong{6, 7};
  const fb::Status status = ring.write_npy(res, dir / "wrong.npy", wrong, span_of(a));
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_FALSE(std::filesystem::exists(dir / "wrong.npy"));
  EXPECT_TRUE(ring.wait());
}

TEST(OutputRing, ReportsAnUnwritablePathAsIO) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::tf::Executor executor{2};
  const fb::test::ScratchDir dir("output_ring");
  const Array a = upload(res, {4}, 0.0);

  fb::OutputRing ring{res.dev_idx(), executor, kTinySlots};
  const fb::Status status =
      ring.write_npy(res, dir / "missing" / "a.npy", a.host.shape, span_of(a));
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::IO);
  EXPECT_TRUE(ring.wait());
}

} // namespace
