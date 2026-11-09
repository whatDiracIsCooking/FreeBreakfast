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
/// The footprint is taken as plain bytes -- its peak, and the allocations live
/// at that peak -- rather than as a `Footprint<K>`: written before that
/// vocabulary (#1044) landed, and a caller passes its peak and live count.

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
};

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
  PoolPeak peak{.granularity = pool_granularity(dev)};
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

/// @brief Hold a measured peak to a footprint of `footprint_peak` bytes with
/// `live_allocs` allocations live at its peak. Two bars, each a non-fatal
/// failure at the caller's line:
///
/// - sound: measured.used <= footprint_peak. Never widened.
/// - tight: footprint_peak - measured.used <= G * live_allocs (+ slack.bytes),
///   G the pool's granularity: each live allocation may round up by under one
///   granule, so a footprint that counts bytes exactly is over by at most that.
inline void expect_footprint(const PoolPeak &measured, std::size_t footprint_peak,
                             std::size_t live_allocs, std::optional<FootprintSlack> slack = {},
                             std::source_location location = std::source_location::current()) {
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
  if (measured.used > footprint_peak) {
    ADD_FAILURE_AT(location.file_name(), location.line())
        << "unsound footprint: measured peak " << measured.used << " bytes exceeds the footprint's "
        << footprint_peak << " by " << (measured.used - footprint_peak) << " bytes";
  } else if (footprint_peak - measured.used > allowance) {
    ADD_FAILURE_AT(location.file_name(), location.line())
        << "loose footprint: " << footprint_peak << " bytes is " << (footprint_peak - measured.used)
        << " over the measured peak " << measured.used << ", past the allowance of " << allowance
        << " (G " << measured.granularity << " x " << live_allocs << " live allocations"
        << (slack ? std::format(" + slack {}", slack->bytes) : std::string{}) << ")";
  }
}

} // namespace free_breakfast::test
