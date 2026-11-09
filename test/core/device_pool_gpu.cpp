// GPU cases for Device's pool cap and pool usage (core/wwr:device), and for the
// test/support helpers built on them (pool_footprint.h): measure_pool_peak and
// expect_footprint, which every footprint test uses.
//
// The cap is maxSize: past it an allocation fails. Through Buffers that failure
// is CreatePolicy's abort, so the cap cases allocate from the pool directly and
// check the error code instead.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
// A cap well above any pool chunk size, so the cap and not the chunking is
// what an allocation runs into.
constexpr std::uint64_t kCap = 256 * kMiB;

void *pool_alloc(fb::Device &dev, std::uint64_t bytes, wwr::wwrError_t &err) {
  void *ptr = nullptr;
  err = wwr::wwrMallocFromPoolAsync(&ptr, bytes, dev.pool(), dev.stream());
  return ptr;
}

std::uint64_t release_threshold(const fb::Device &dev) {
  std::uint64_t value = 0;
  EXPECT_EQ(wwr::wwrMemPoolGetAttribute(dev.pool(), wwr::wwrMemPoolAttrReleaseThreshold, &value),
            wwr::wwrSuccess);
  return value;
}

// Under a cap of C, C/2 fits and a second allocation that would take the pool
// past C fails with an allocation error. The failure is cleared from the
// runtime's last-error slot, so it does not leak into a later check.
TEST(DevicePool, CapRefusesAnAllocationPastIt) {
  FBF_REQUIRE_GPU();

  fb::Device dev{0, fb::Device::kDefaultPoolReleaseThreshold, kCap};

  wwr::wwrError_t err{};
  void *half = pool_alloc(dev, kCap / 2, err);
  ASSERT_EQ(err, wwr::wwrSuccess);
  ASSERT_NE(half, nullptr);

  void *past = pool_alloc(dev, kCap / 2 + kMiB, err);
  EXPECT_EQ(err, wwr::wwrErrorMemoryAllocation);
  (void)wwr::wwrGetLastError();
  if (err == wwr::wwrSuccess) {
    EXPECT_EQ(wwr::wwrFreeAsync(past, dev.stream()), wwr::wwrSuccess);
  }

  EXPECT_EQ(wwr::wwrFreeAsync(half, dev.stream()), wwr::wwrSuccess);
  EXPECT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
}

// A cap is also the release threshold: memory freed under it stays reserved
// in the pool across a stream sync, for the next step to reuse.
TEST(DevicePool, CapKeepsFreedMemoryAcrossASync) {
  FBF_REQUIRE_GPU();

  fb::Device dev{0, fb::Device::kDefaultPoolReleaseThreshold, kCap};
  EXPECT_EQ(release_threshold(dev), kCap);

  wwr::wwrError_t err{};
  void *ptr = pool_alloc(dev, kCap / 2, err);
  ASSERT_EQ(err, wwr::wwrSuccess);
  ASSERT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
  const auto held = dev.pool_usage();
  EXPECT_GE(held.used_current, kCap / 2);
  EXPECT_GE(held.reserved_current, held.used_current);

  ASSERT_EQ(wwr::wwrFreeAsync(ptr, dev.stream()), wwr::wwrSuccess);
  ASSERT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
  const auto freed = dev.pool_usage();
  EXPECT_EQ(freed.used_current, 0U);
  EXPECT_EQ(freed.reserved_current, held.reserved_current);
}

// Uncapped, the threshold is the one passed (kDefaultPoolReleaseThreshold by
// default), and a threshold of 0 still hands freed memory back at the sync.
TEST(DevicePool, UncappedKeepsTheReleaseThreshold) {
  FBF_REQUIRE_GPU();

  EXPECT_EQ(release_threshold(fb::Device{0}), fb::Device::kDefaultPoolReleaseThreshold);

  fb::Device dev{0, 0};
  EXPECT_EQ(release_threshold(dev), 0U);
  wwr::wwrError_t err{};
  void *ptr = pool_alloc(dev, 64 * kMiB, err);
  ASSERT_EQ(err, wwr::wwrSuccess);
  ASSERT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
  EXPECT_GT(dev.pool_usage().reserved_current, 0U);

  ASSERT_EQ(wwr::wwrFreeAsync(ptr, dev.stream()), wwr::wwrSuccess);
  ASSERT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
  EXPECT_EQ(dev.pool_usage().reserved_current, 0U);
}

// The high-water marks follow the peak, and a reset brings them back to the
// current values.
TEST(DevicePool, HighWaterResetsToCurrent) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  {
    fb::Buffers::device<double> big(std::size_t{1} << 20, dev);
    ASSERT_EQ(wwr::extension::sync(dev->stream()), wwr::wwrSuccess);
  }
  ASSERT_EQ(wwr::extension::sync(dev->stream()), wwr::wwrSuccess);
  const auto before = dev->pool_usage();
  EXPECT_EQ(before.used_current, 0U);
  EXPECT_GE(before.used_high, std::uint64_t{8} << 20);

  dev->reset_pool_high_water();
  const auto after = dev->pool_usage();
  EXPECT_EQ(after.used_high, after.used_current);
  EXPECT_EQ(after.reserved_high, after.reserved_current);
}

// One known Buffers::device<double>(n): the peak is n * 8 rounded up to the
// pool's granularity, and expect_footprint holds its two bars around it.
TEST(DevicePool, PeakOfOneBufferAndItsFootprintBars) {
  FBF_REQUIRE_GPU();

  // Not a multiple of any plausible granularity, so the rounding shows.
  constexpr std::size_t kN = 1000;
  auto dev = std::make_shared<fb::Device>(0);
  const fbt::PoolPeak peak =
      fbt::measure_pool_peak(*dev, [&] { fb::Buffers::device<double> d(kN, dev); });

  const std::size_t g = peak.granularity;
  ASSERT_GT(g, 0U);
  const std::size_t exact = fb::math::align_up(kN * sizeof(double), g);
  EXPECT_EQ(peak.used, exact);

  // A footprint of `bytes`, all of it scratch: the buffer lives only inside
  // the measured call.
  const auto fp = [](std::size_t bytes) {
    return fb::Footprint<wwr::extension::MemoryKind::Device>{.scratch = bytes};
  };
  const fbt::FootprintSlack granule{.bytes = g, .reason = "one granule, to test the widening"};

  fbt::expect_footprint(peak, fp(exact), 1);
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(exact - g), 1), "unsound footprint");
  // A slack widens the tight bar only: one granule under is still unsound.
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(exact - g), 1, granule),
                          "unsound footprint");
  fbt::expect_footprint(peak, fp(exact + g), 1, granule);
  // Past G x A, the tight bar fails without the slack and passes with it.
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(exact + 2 * g), 1), "loose footprint");
  fbt::expect_footprint(peak, fp(exact + 2 * g), 1, granule);
  // A slack without a reason is refused.
  EXPECT_NONFATAL_FAILURE(
      fbt::expect_footprint(peak, fp(exact), 1, fbt::FootprintSlack{.bytes = g}), "needs a reason");
}

} // namespace
