/// @file
/// @brief Measure one call's peak pool usage on a Device, and hold it to a
/// footprint: the two bars every footprint test states.
///
/// Lives in test/support/ (target free_breakfast::test::support), the one home
/// of the helpers every gtest tree shares; consumers include it by bare name.
///
/// Include it AFTER gtest.h, `import std;` and `import free_breakfast.core;`
/// (or `free_breakfast.core.wwr`), like scratch_dir.h: it includes no standard
/// header itself and names `std::`, `wwr::` and `free_breakfast::Device` only
/// through the including TU's imports, so the header build and the module
/// build never overlap.
///
/// The footprint held to a measurement is a device `Footprint<K>`
/// (free_breakfast.core.footprint, re-exported by free_breakfast.core): the
/// pool measured is a device pool.

#pragma once

namespace free_breakfast::test {

/// @brief One call's pool usage above where it started, in bytes.
struct PoolPeak {
  /// UsedMemHigh during the call, less UsedMemCurrent at its start: the most
  /// the call's own allocations held at once, each rounded to `granularity`.
  std::size_t used = 0;
  /// ReservedMemHigh during the call, less ReservedMemCurrent at its start:
  /// what the pool had to take from the driver for it (0 when its cache
  /// already held enough).
  std::size_t reserved = 0;
  /// The pool's allocation granularity G: what a one-byte allocation adds to
  /// UsedMemCurrent, measured on the same pool just before the call.
  std::size_t granularity = 0;
  /// Whether the pool rounds a reused block (pool_rounds_reuse): ROCm's does,
  /// CUDA's does not.
  bool reuse_rounding = false;
};

/// @brief The bound on a pool's reuse rounding, as a divisor of the request:
/// ROCm 7.2.4's pool serves a request whole from a freed block of up to 9/8
/// of its size, and UsedMem counts the block, so one allocation is counted at
/// most request / 8 past what it asked (src/core/wwr/README.md, measured by
/// test/core/device_pool_gpu.cpp's DevicePool.ReuseRounding). CUDA's counts
/// the request.
inline constexpr std::size_t kPoolReuseDivisor = 8;

/// @brief Whether the pool of device @p dev_idx counts a reused block whole
/// rather than the request: a 1024-byte block freed, then 1000 bytes asked
/// for, on a throwaway Device, so the measured pool's cache is untouched.
/// Measured once per process.
inline bool pool_rounds_reuse(std::int32_t dev_idx) {
  static const bool rounds = [dev_idx] {
    Device dev{dev_idx};
    void *ptr = nullptr;
    EXPECT_EQ(wwr::wwrMallocFromPoolAsync(&ptr, 1024, dev.pool(), dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrFreeAsync(ptr, dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    const auto before = dev.pool_usage().used_current;
    EXPECT_EQ(wwr::wwrMallocFromPoolAsync(&ptr, 1000, dev.pool(), dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    const auto grown = dev.pool_usage().used_current - before;
    EXPECT_EQ(wwr::wwrFreeAsync(ptr, dev.stream()), wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
    return grown == 1024;
  }();
  return rounds;
}

/// @brief The pool's allocation granularity: the UsedMemCurrent a one-byte
/// allocation adds, measured. Neither backend states it for a pool, so it is
/// read off rather than assumed.
inline std::size_t pool_granularity(Device &dev) {
  const auto before = dev.pool_usage().used_current;
  void *ptr = nullptr;
  EXPECT_EQ(wwr::wwrMallocFromPoolAsync(&ptr, 1, dev.pool(), dev.stream()), wwr::wwrSuccess);
  const auto after = dev.pool_usage().used_current;
  EXPECT_EQ(wwr::wwrFreeAsync(ptr, dev.stream()), wwr::wwrSuccess);
  EXPECT_EQ(wwr::extension::sync(dev.stream()), wwr::wwrSuccess);
  return static_cast<std::size_t>(after - before);
}

/// @brief Run `call` and return its peak pool usage on `dev`: reset the
/// high-water marks, run, synchronize the device, read them back.
///
/// The device sync, not just dev.stream()'s, so work `call` forks onto other
/// streams of the same pool has finished too.
template <typename Call>
PoolPeak measure_pool_peak(Device &dev, Call &&call) {
  PoolPeak peak{.granularity = pool_granularity(dev),
                .reuse_rounding = pool_rounds_reuse(dev.dev_idx())};
  dev.reset_pool_high_water();
  const auto start = dev.pool_usage();
  std::forward<Call>(call)();
  EXPECT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
  const auto end = dev.pool_usage();
  peak.used = static_cast<std::size_t>(end.used_high - start.used_current);
  peak.reserved = static_cast<std::size_t>(end.reserved_high - start.reserved_current);
  return peak;
}

/// @brief A widening of the tight bar, for a pool allocation in the call tree
/// that the footprint deliberately leaves out (a library temporary that goes
/// through the pool, say). It needs a reason, and the reason is logged.
struct FootprintSlack {
  std::size_t bytes = 0;
  std::string_view reason;
};

/// @brief Hold a measured peak to `footprint`, a step's or a library
/// routine's, with `live_allocs` allocations live at its peak. Two bars, each a non-fatal
/// failure at the caller's line:
///
/// - sound: measured.used <= footprint.peak(), on a pool that counts requests
///   (CUDA's). On one that rounds reuse (ROCm's, measured.reuse_rounding),
///   measured.used <= footprint.peak() * (1 + 1 / kPoolReuseDivisor): each
///   allocation live at the peak may be served from a freed block and counted
///   at most request / 8 past what it asked, and the live allocations' requests
///   sum to at most the footprint. That is pool rounding, as G is, and the
///   footprint keeps counting requested bytes; at run time pre-flight's 64 MiB
///   of headroom covers the overcounts measured (src/pipeline/footprint/
///   README.md says where the bound outgrows it). Logged whenever
///   the allowance is used. Never widened otherwise.
/// - tight: footprint.peak() - measured.used <= G * live_allocs (+ slack.bytes),
///   G the pool's granularity: each live allocation may round up by under one
///   granule, so a footprint that counts bytes exactly is over by at most that.
inline void expect_footprint(const PoolPeak &measured,
                             const Footprint<wwr::extension::MemoryKind::Device> &footprint,
                             std::size_t live_allocs, std::optional<FootprintSlack> slack = {},
                             std::source_location location = std::source_location::current()) {
  const std::size_t footprint_peak = footprint.peak();
  const std::size_t rounding = measured.granularity * live_allocs;
  std::size_t allowance = rounding;
  if (slack) {
    if (slack->reason.empty()) {
      ADD_FAILURE_AT(location.file_name(), location.line())
          << "a footprint slack needs a reason (" << slack->bytes << " bytes given)";
    }
    std::cout << std::format("[ footprint ] slack of {} bytes: {}\n", slack->bytes, slack->reason);
    allowance += slack->bytes;
  }
  const std::size_t reuse = measured.reuse_rounding ? footprint_peak / kPoolReuseDivisor : 0;
  if (measured.used > footprint_peak + reuse) {
    ADD_FAILURE_AT(location.file_name(), location.line())
        << "unsound footprint: measured peak " << measured.used << " bytes exceeds the footprint's "
        << footprint_peak << " by " << (measured.used - footprint_peak) << " bytes"
        << (reuse != 0 ? std::format(", past the pool's reuse rounding of {}", reuse)
                       : std::string{});
  } else if (measured.used > footprint_peak) {
    std::cout << std::format("[ footprint ] reuse rounding: measured {} is {} over the footprint's "
                             "{}, within the pool's {} (a freed block served whole)\n",
                             measured.used, measured.used - footprint_peak, footprint_peak, reuse);
  } else if (footprint_peak - measured.used > allowance) {
    ADD_FAILURE_AT(location.file_name(), location.line())
        << "loose footprint: " << footprint_peak << " bytes is " << (footprint_peak - measured.used)
        << " over the measured peak " << measured.used << ", past the allowance of " << allowance
        << " (G " << measured.granularity << " x " << live_allocs << " live allocations"
        << (slack ? std::format(" + slack {}", slack->bytes) : std::string{}) << ")";
  }
}

} // namespace free_breakfast::test
