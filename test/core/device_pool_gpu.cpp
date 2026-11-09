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
  // Under: one granule, or on a pool that rounds reuse (ROCm's), past its
  // allowance of footprint / kPoolReuseDivisor, within which one granule
  // under passes.
  const std::size_t under =
      peak.reuse_rounding ? ((exact * fbt::kPoolReuseDivisor) / (fbt::kPoolReuseDivisor + 1)) - 1
                          : exact - g;
  if (peak.reuse_rounding) {
    fbt::expect_footprint(peak, fp(exact - g), 1);
  }
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(under), 1), "unsound footprint");
  // A slack widens the tight bar only: under is still unsound.
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(under), 1, granule), "unsound footprint");
  fbt::expect_footprint(peak, fp(exact + g), 1, granule);
  // Past G x A, the tight bar fails without the slack and passes with it.
  EXPECT_NONFATAL_FAILURE(fbt::expect_footprint(peak, fp(exact + 2 * g), 1), "loose footprint");
  fbt::expect_footprint(peak, fp(exact + 2 * g), 1, granule);
  // A slack without a reason is refused.
  EXPECT_NONFATAL_FAILURE(
      fbt::expect_footprint(peak, fp(exact), 1, fbt::FootprintSlack{.bytes = g}), "needs a reason");
}

// The pool's reuse rounding, the bound pool_footprint.h's sound bar allows on
// a pool that has it: a block X freed and then reused, whole, for a smaller
// request Y, UsedMemCurrent counting X. For each X from 2 B to 64 MiB, on a
// fresh Device so nothing else is cached, the smallest Y the pool serves from
// X, by bisection. On ROCm 7.2.4 (gfx1200) it is the smallest Y with
// 8 X <= 9 Y, so X - Y <= Y / 8 (kPoolReuseDivisor), reached from ~12 kB up;
// CUDA's pool counts the request, and serves nothing whole.
TEST(DevicePool, ReuseRounding) {
  FBF_REQUIRE_GPU();
  // Whether a request of y, after a block of x is freed, is counted as x.
  const auto reuses = [](std::uint64_t x, std::uint64_t y) {
    fb::Device dev{0};
    wwr::wwrError_t err{};
    void *first = pool_alloc(dev, x, err);
    EXPECT_EQ(err, wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrFreeAsync(first, dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    const std::uint64_t before = dev.pool_usage().used_current;
    void *second = pool_alloc(dev, y, err);
    EXPECT_EQ(err, wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    const std::uint64_t grown = dev.pool_usage().used_current - before;
    EXPECT_EQ(wwr::wwrFreeAsync(second, dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    EXPECT_TRUE(grown == y || grown == x) << x << " " << y << " " << grown;
    return grown == x && x != y;
  };
  std::uint64_t most = 0;
  double ratio = 0.0;
  for (std::uint64_t x = 2; x <= (std::uint64_t{64} << 20U); x = (x * 3) / 2 + 1) {
    if (!reuses(x, x - 1)) {
      continue;
    }
    std::uint64_t lo = 1;     // not reused (or x itself)
    std::uint64_t hi = x - 1; // reused
    if (reuses(x, lo)) {
      hi = lo;
    }
    while (hi - lo > 1) {
      const std::uint64_t mid = lo + ((hi - lo) / 2);
      (reuses(x, mid) ? hi : lo) = mid;
    }
    const std::uint64_t over = x - hi;
    EXPECT_LE(over * fbt::kPoolReuseDivisor, hi) << "block " << x;
    most = std::max(most, over);
    ratio = std::max(ratio, static_cast<double>(over) / static_cast<double>(hi));
    std::cout << std::format(
        "[ reuse ] block {} smallest request served {} overcount {} ({:.4f})\n", x, hi, over,
        static_cast<double>(over) / static_cast<double>(hi));
  }
  std::cout << std::format("[ reuse ] largest overcount {}, largest ratio {:.4f}\n", most, ratio);
  EXPECT_EQ(most > 0, fbt::pool_rounds_reuse(0));
}

} // namespace
