// GPU cases for core/gpu: the wwr wrappers, bound to this build's policies,
// constructed inside gpu::capture on a real device.
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
namespace gpu = free_breakfast::gpu;
namespace kit = wwr::extension::kit;

constexpr std::size_t kCount = 1024;

TEST(Gpu, WrappersConstructInsideCapture) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto r = gpu::capture([] {
    gpu::ScopedRange range{"gpu_policy_gpu"};
    auto dev = std::make_shared<gpu::Device>(0);
    gpu::Buffers::device<float> d(kCount, dev);
    gpu::Buffers::host<float> h(kCount);
    gpu::BlasHandle blas(dev);
    gpu::SolverHandle solver(dev);
    return std::tuple{d.num_elements(), h.num_elements(), blas.valid(), solver.valid()};
  });
  ASSERT_TRUE(r.has_value()) << r.error().message;
  auto [d_n, h_n, blas_ok, solver_ok] = *r;
  EXPECT_EQ(d_n, kCount);
  EXPECT_EQ(h_n, kCount);
  EXPECT_TRUE(blas_ok);
  EXPECT_TRUE(solver_ok);
}

TEST(Gpu, DeviceAndBufferReturnThroughCapture) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto dev = gpu::capture([] { return std::make_shared<gpu::Device>(0); });
  ASSERT_TRUE(dev.has_value()) << dev.error().message;

  auto buf = gpu::capture([&] { return gpu::Buffers::device<double>(kCount, *dev); });
  ASSERT_TRUE(buf.has_value()) << buf.error().message;
  EXPECT_EQ(buf->num_elements(), kCount);
  EXPECT_NE(buf->data(), nullptr);
}

// Named explicitly, not through gpu::PolicyMap, so this case checks the
// RECORD path in every build -- including a Debug one, where PolicyMap aborts.
struct RecordMap {
  template<typename E>
  using alloc = gpu::RecordPolicy<E>;
  template<typename E>
  using free = gpu::LogPolicy<E>;
};
using RecordingBuffers = kit::device_buffer_suite<RecordMap, gpu::Device>;

TEST(Gpu, FailedAllocationComesBackAsError) {
  FREE_BREAKFAST_REQUIRE_GPU();

  auto dev = gpu::capture([] { return std::make_shared<gpu::Device>(0); });
  ASSERT_TRUE(dev.has_value()) << dev.error().message;

  constexpr auto too_many = std::numeric_limits<std::size_t>::max();
  int line = 0;
  auto buf = gpu::capture([&] {
    line = std::source_location::current().line() + 1;
    return RecordingBuffers::device<float>(too_many, *dev);
  });
  ASSERT_FALSE(buf.has_value());
  EXPECT_EQ(buf.error().kind, fb::ErrorKind::GpuRuntime);
  EXPECT_EQ(buf.error().origin.line(), static_cast<std::uint_least32_t>(line));
}

} // namespace
