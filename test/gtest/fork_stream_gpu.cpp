// GPU cases for core/wwr's :fork_stream: a ForkStream constructed on a real
// device, the fork/join protocol ordering work against a parent stream, and
// the same protocol inside a graph capture. The static_asserts at the top
// are checked whenever this binary builds, device or not.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

// Owns three move-only wrappers: movable, not copyable.
static_assert(std::is_move_constructible_v<fb::ForkStream>);
static_assert(std::is_move_assignable_v<fb::ForkStream>);
static_assert(!std::is_copy_constructible_v<fb::ForkStream>);
static_assert(!std::is_copy_assignable_v<fb::ForkStream>);

// dev_idx() + stream(): the stream tier of wwr's device_handle ladder, so a
// buffer suite can be bound to a ForkStream the way it can to a bare Stream.
static_assert(ext::device_handle_stream<fb::ForkStream>);
static_assert(!ext::device_handle_pool<fb::ForkStream>);

constexpr std::size_t kCount = 1024;

TEST(ForkStream, ConstructsOnTheDevice) {
  FBF_REQUIRE_GPU();

  fb::ForkStream fs{0};
  EXPECT_EQ(fs.dev_idx(), 0);
  EXPECT_TRUE(fs.stream().valid());
  EXPECT_TRUE(fs.fork_event().valid());
  EXPECT_TRUE(fs.join_event().valid());
  EXPECT_EQ(fs.fork_event().dev_idx(), 0);
  EXPECT_EQ(fs.join_event().dev_idx(), 0);
  EXPECT_NE(fs.fork_event().get(), fs.join_event().get());
}

// The priority path takes the (flags, priority) stream constructor; the
// runtime clamps a priority into its range, so any value is accepted.
TEST(ForkStream, ConstructsFromConfig) {
  FBF_REQUIRE_GPU();

  std::int32_t least = 0;
  std::int32_t greatest = 0;
  ASSERT_EQ(wwr::wwrDeviceGetStreamPriorityRange(&least, &greatest), wwr::wwrSuccess);

  fb::ForkStream fs{fb::ForkStreamConfig{
      .dev_idx = 0,
      .stream_flags = wwr::wwrStreamNonBlocking,
      .stream_priority = greatest,
      .event_flags = 0, // timing-capable
  }};
  EXPECT_TRUE(fs.stream().valid());
  EXPECT_TRUE(fs.fork_event().valid());
  EXPECT_TRUE(fs.join_event().valid());

  std::int32_t priority = 0;
  ASSERT_EQ(wwr::wwrStreamGetPriority(fs.stream(), &priority), wwr::wwrSuccess);
  EXPECT_EQ(priority, greatest);
}

// Fork off the Device's stream, fill a buffer on the fork, join back, and read
// it on the parent: the read sees the fork's writes only if join_into ordered it.
TEST(ForkStream, ForkJoinOrdersWorkAgainstTheParent) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::Stream &parent = dev->stream();
  fb::ForkStream side{0};

  fb::Buffers::device<float> d(kCount, dev);
  fb::Buffers::pinned<float> in(kCount);
  fb::Buffers::pinned<float> out(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    in.data()[i] = static_cast<float>(i);
    out.data()[i] = -1.0F;
  }

  ASSERT_EQ(ext::memset(d, 0, parent), wwr::wwrSuccess);

  ASSERT_EQ(side.fork_from(parent), wwr::wwrSuccess);
  ASSERT_EQ(ext::copy(d, in, side.stream()), wwr::wwrSuccess);
  ASSERT_EQ(side.record_join(), wwr::wwrSuccess);
  ASSERT_EQ(side.join_into(parent), wwr::wwrSuccess);

  ASSERT_EQ(ext::copy(out, d, parent), wwr::wwrSuccess);
  ASSERT_EQ(ext::sync(parent), wwr::wwrSuccess);

  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(out.data()[i], static_cast<float>(i)) << "at " << i;
  }

  ASSERT_EQ(side.sync(), wwr::wwrSuccess);
}

// The protocol inside a capture: fork_from joins the fork to the parent's
// capture, join_into brings it back, and the instantiated graph carries the
// fork's copy. end_capture would fail with an unjoined fork.
//
// Everything lives in an inner scope so the destructors run inside the test:
// an event recorded only during a capture cannot be synchronized afterwards,
// and Event's destroy synchronizes first. Without ForkStream's re-record that
// path logs two errors and leaves one as the runtime's last error, which the
// next test's wwrGetLastError() would then report as its own -- the
// single-process smoke run caught exactly that. The final check pins it.
TEST(ForkStream, ForkJoinInsideGraphCapture) {
  FBF_REQUIRE_GPU();

  {
    auto dev = std::make_shared<fb::Device>(0);
    fb::Stream &parent = dev->stream();
    fb::ForkStream side{0};

    fb::Buffers::device<float> d(kCount, dev);
    fb::Buffers::pinned<float> in(kCount);
    fb::Buffers::pinned<float> out(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
      in.data()[i] = static_cast<float>(2 * i);
      out.data()[i] = -1.0F;
    }

    ASSERT_EQ(ext::begin_capture(parent), wwr::wwrSuccess);
    ASSERT_EQ(side.fork_from(parent), wwr::wwrSuccess);
    ASSERT_EQ(ext::copy(d, in, side.stream()), wwr::wwrSuccess);
    ASSERT_EQ(side.record_join(), wwr::wwrSuccess);
    ASSERT_EQ(side.join_into(parent), wwr::wwrSuccess);
    ASSERT_EQ(ext::copy(out, d, parent), wwr::wwrSuccess);
    auto graph = parent.end_capture();
    ASSERT_TRUE(graph.valid());

    // Nothing has run yet: capture records, it does not execute.
    ASSERT_EQ(ext::sync(parent), wwr::wwrSuccess);
    EXPECT_EQ(out.data()[0], -1.0F);

    auto exec = graph.instantiate();
    ASSERT_TRUE(exec.valid());
    ASSERT_EQ(ext::launch(exec, parent), wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(parent), wwr::wwrSuccess);

    for (std::size_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(out.data()[i], static_cast<float>(2 * i)) << "at " << i;
    }
  }

  EXPECT_EQ(wwr::wwrGetLastError(), wwr::wwrSuccess);
}

// Moving a ForkStream whose events were recorded in a capture: the destination
// takes them over and re-records them on the way out, as the destructor does.
TEST(ForkStream, MoveAfterCaptureLeavesNoError) {
  FBF_REQUIRE_GPU();

  {
    auto dev = std::make_shared<fb::Device>(0);
    fb::Stream &parent = dev->stream();
    fb::ForkStream side{0};
    fb::ForkStream other{0};

    ASSERT_EQ(ext::begin_capture(parent), wwr::wwrSuccess);
    ASSERT_EQ(side.fork_from(parent), wwr::wwrSuccess);
    ASSERT_EQ(side.record_join(), wwr::wwrSuccess);
    ASSERT_EQ(side.join_into(parent), wwr::wwrSuccess);
    auto graph = parent.end_capture();
    ASSERT_TRUE(graph.valid());

    // Move-construct, then move-assign over a live ForkStream: both paths
    // must leave the runtime clean.
    fb::ForkStream moved{std::move(side)};
    other = std::move(moved);
    EXPECT_TRUE(other.stream().valid());
  }

  EXPECT_EQ(wwr::wwrGetLastError(), wwr::wwrSuccess);
}

// Moving hands over the stream and both events; the source keeps nothing.
TEST(ForkStream, MoveTransfersOwnership) {
  FBF_REQUIRE_GPU();

  fb::ForkStream src{0};
  const wwr::wwrStream_t stream = src.stream();
  const wwr::wwrEvent_t fork = src.fork_event();
  const wwr::wwrEvent_t join = src.join_event();

  fb::ForkStream dst{std::move(src)};

  EXPECT_EQ(dst.dev_idx(), 0);
  EXPECT_EQ(dst.stream().get(), stream);
  EXPECT_EQ(dst.fork_event().get(), fork);
  EXPECT_EQ(dst.join_event().get(), join);

  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is the point.
  EXPECT_EQ(src.dev_idx(), -1);
  EXPECT_FALSE(src.stream().valid());
  EXPECT_FALSE(src.fork_event().valid());
  EXPECT_FALSE(src.join_event().valid());
}

} // namespace
