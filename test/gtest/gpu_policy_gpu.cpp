// GPU cases for core/gpu: the wwr wrappers, bound to this build's policies,
// constructed inside fb::capture on a real device.
//
// The failure case uses the size-overflow rejection (a count past
// max_num_elements) rather than a real out-of-memory allocation: it reaches the
// alloc policy the same way, but allocates nothing, so it is safe under ASan and
// compute-sanitizer -- both of which treat a genuine failed allocation as an
// error of their own.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

namespace fb = free_breakfast;
namespace kit = wwr::extension::kit;

constexpr std::size_t kCount = 1024;

TEST(Gpu, WrappersConstructInsideCapture) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto r = fb::capture([] {
    fb::ScopedRange range{"gpu_policy_gpu"};
    auto dev = std::make_shared<fb::Device>(0);
    fb::Buffers::device<float> d(kCount, dev);
    fb::Buffers::host<float> h(kCount);
    fb::BlasHandle blas(dev);
    fb::SolverHandle solver(dev);
    fb::Event event;       // default: sync-only, timing disabled
    fb::Event timed(0, 0); // flags = 0: timing-capable
    return std::tuple{d.num_elements(), h.num_elements(), blas.valid(),
                      solver.valid(),   event.valid(),    timed.valid()};
  });
  ASSERT_TRUE(r.has_value()) << r.error().message;
  auto [d_n, h_n, blas_ok, solver_ok, event_ok, timed_ok] = *r;
  EXPECT_EQ(d_n, kCount);
  EXPECT_EQ(h_n, kCount);
  EXPECT_TRUE(blas_ok);
  EXPECT_TRUE(solver_ok);
  EXPECT_TRUE(event_ok);
  EXPECT_TRUE(timed_ok);
}

TEST(Gpu, DeviceAndBufferReturnThroughCapture) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto dev = fb::capture([] { return std::make_shared<fb::Device>(0); });
  ASSERT_TRUE(dev.has_value()) << dev.error().message;

  auto buf = fb::capture([&] { return fb::Buffers::device<double>(kCount, *dev); });
  ASSERT_TRUE(buf.has_value()) << buf.error().message;
  EXPECT_EQ(buf->num_elements(), kCount);
  EXPECT_NE(buf->data(), nullptr);
}

// Named explicitly, not through fb::PolicyMap, so this case checks the
// RECORD path in every build -- including a Debug one, where PolicyMap aborts.
struct RecordMap {
  template <typename E>
  using alloc = fb::RecordPolicy<E>;
  template <typename E>
  using free = fb::LogPolicy<E>;
};
using RecordingBuffers = kit::device_buffer_suite<RecordMap, fb::Device>;

TEST(Gpu, FailedAllocationComesBackAsError) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto dev = fb::capture([] { return std::make_shared<fb::Device>(0); });
  ASSERT_TRUE(dev.has_value()) << dev.error().message;

  constexpr auto too_many = std::numeric_limits<std::size_t>::max();
  int line = 0;
  auto buf = fb::capture([&] {
    line = std::source_location::current().line() + 1;
    return RecordingBuffers::device<float>(too_many, *dev);
  });
  ASSERT_FALSE(buf.has_value());
  EXPECT_EQ(buf.error().kind, fb::ErrorKind::GpuRuntime);
  EXPECT_EQ(buf.error().origin.line(), static_cast<std::uint_least32_t>(line));
}

} // namespace
