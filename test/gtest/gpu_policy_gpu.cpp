// GPU cases for core/wwr: the wwr wrappers, bound to this tree's policies,
// constructed on a real device. A failed creation aborts (AbortPolicy); that
// path is covered on the host, in gpu_policy_host.cpp.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;

constexpr std::size_t kCount = 1024;

TEST(Gpu, WrappersConstruct) {
  FBF_REQUIRE_GPU();

  fb::ScopedRange range{"gpu_policy_gpu"};
  auto dev = std::make_shared<fb::Device>(0);
  fb::Buffers::device<float> d(kCount, dev);
  fb::Buffers::host<float> h(kCount);
  fb::BlasHandle blas(dev);
  fb::SolverHandle solver(dev);
  fb::Event event;       // default: sync-only, timing disabled
  fb::Event timed(0, 0); // flags = 0: timing-capable

  EXPECT_EQ(d.num_elements(), kCount);
  EXPECT_NE(d.data(), nullptr);
  EXPECT_EQ(h.num_elements(), kCount);
  EXPECT_TRUE(blas.valid());
  EXPECT_TRUE(solver.valid());
  EXPECT_TRUE(event.valid());
  EXPECT_TRUE(timed.valid());
}

} // namespace
