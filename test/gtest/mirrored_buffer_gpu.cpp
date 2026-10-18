// GPU cases for core/mirrored_buffer: round trips through the device side with
// the _sync and async copies, and destruction with a copy still in flight. The
// static_asserts at the top are checked whenever this binary builds, device or
// not.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

// Owns two move-only buffers: movable, not copyable.
static_assert(std::is_move_constructible_v<fb::MirroredBuffer<double>>);
static_assert(std::is_move_assignable_v<fb::MirroredBuffer<double>>);
static_assert(!std::is_copy_constructible_v<fb::MirroredBuffer<double>>);
static_assert(!std::is_copy_assignable_v<fb::MirroredBuffer<double>>);

constexpr std::size_t kCount = 1024;

void fill(fb::MirroredBuffer<double> &m, double offset) {
  for (std::size_t i = 0; i < m.size(); ++i) {
    m.pinned().data()[i] = offset + static_cast<double>(i);
  }
}

TEST(MirroredBuffer, ConstructsZeroedAndEqualSized) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::MirroredBuffer<double> m{kCount, dev};
  EXPECT_EQ(m.size(), kCount);
  EXPECT_EQ(m.device().num_elements(), kCount);
  EXPECT_EQ(m.pinned().num_elements(), kCount);

  // The device side's zero-fill is queued on the same stream, so the d2h is
  // ordered after it.
  fill(m, 1.0);
  ASSERT_EQ(m.to_pinned_sync(), wwr::wwrSuccess);
  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(m.pinned().data()[i], 0.0) << "at " << i;
  }
}

// h2d, clobber the host side, d2h: the values come back only by way of the
// device side.
TEST(MirroredBuffer, SyncRoundTrip) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::MirroredBuffer<double> m{kCount, dev};

  fill(m, 0.5);
  ASSERT_EQ(m.to_device_sync(), wwr::wwrSuccess);
  std::ranges::fill(std::span{m.pinned().data(), m.size()}, -1.0);
  ASSERT_EQ(m.to_pinned_sync(), wwr::wwrSuccess);

  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(m.pinned().data()[i], 0.5 + static_cast<double>(i)) << "at " << i;
  }
}

// Both copies enqueued back to back on the work stream, then one sync: stream
// order alone keeps the d2h after the h2d. The device side is cleared in
// between, on the same stream, so a d2h that skipped the h2d would read zeros.
TEST(MirroredBuffer, AsyncRoundTripOrderedByTheStream) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::MirroredBuffer<double> m{kCount, dev};
  fb::MirroredBuffer<double> out{kCount, dev};

  fill(m, 2.0);
  ASSERT_EQ(m.to_device(), wwr::wwrSuccess);
  ASSERT_EQ(ext::memset(out.device(), 0, dev->stream()), wwr::wwrSuccess);
  ASSERT_EQ(ext::copy(out.device(), m.device(), dev->stream()), wwr::wwrSuccess);
  ASSERT_EQ(out.to_pinned(), wwr::wwrSuccess);
  ASSERT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);

  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(out.pinned().data()[i], 2.0 + static_cast<double>(i)) << "at " << i;
  }
}

// A d2h left in flight at destruction: the destructor's sync must drain it
// before the pinned side is freed. Under compute-sanitizer a missing sync shows
// as an access to freed host memory; here, the check is that destruction
// leaves no error behind for the next call to report as its own.
TEST(MirroredBuffer, DestroysWithACopyInFlight) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  {
    fb::MirroredBuffer<double> m{kCount, dev};
    fill(m, 3.0);
    ASSERT_EQ(m.to_device(), wwr::wwrSuccess);
    ASSERT_EQ(m.to_pinned(), wwr::wwrSuccess);
  }
  EXPECT_EQ(wwr::wwrGetLastError(), wwr::wwrSuccess);
}

// Move construction carries both sides; move assignment releases the target's
// own buffers (after syncing) and takes the source's.
TEST(MirroredBuffer, MovesCarryBothSides) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::MirroredBuffer<double> a{kCount, dev};
  fill(a, 4.0);
  ASSERT_EQ(a.to_device_sync(), wwr::wwrSuccess);

  fb::MirroredBuffer<double> b{std::move(a)};
  EXPECT_EQ(b.size(), kCount);

  fb::MirroredBuffer<double> c{kCount / 2, dev};
  c = std::move(b);
  EXPECT_EQ(c.size(), kCount);

  std::ranges::fill(std::span{c.pinned().data(), c.size()}, -1.0);
  ASSERT_EQ(c.to_pinned_sync(), wwr::wwrSuccess);
  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(c.pinned().data()[i], 4.0 + static_cast<double>(i)) << "at " << i;
  }
}

} // namespace
