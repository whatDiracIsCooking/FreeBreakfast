// GPU smoke cases for core/device_resources: DeviceResources constructed on a
// real device, its handles bound to the Device's stream, and the one-way
// ownership that lets a buffer outlive the bundle that handed out its Device.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::size_t kCount = 1024;

// Both handles live on the Device's device and are bound to its one stream.
TEST(DeviceResources, HandlesShareTheDeviceStream) {
  FBF_REQUIRE_GPU();

  const fb::DeviceResources res{0};

  ASSERT_NE(res.device(), nullptr);
  EXPECT_EQ(res.dev_idx(), 0);
  EXPECT_NE(res.stream(), nullptr);

  EXPECT_TRUE(res.blas().valid());
  EXPECT_EQ(res.blas().dev_idx(), res.dev_idx());
  EXPECT_EQ(res.blas().stream(), res.stream());

  EXPECT_TRUE(res.solver().valid());
  EXPECT_EQ(res.solver().dev_idx(), res.dev_idx());
  EXPECT_EQ(res.solver().stream(), res.stream());
}

// A buffer drawn from device() retains the Device, so it stays usable -- its
// pool and stream alive -- after the DeviceResources is gone.
TEST(DeviceResources, BufferOutlivesResources) {
  FBF_REQUIRE_GPU();

  std::shared_ptr<fb::Device> dev;
  std::optional<fb::Buffers::device<float>> d;
  {
    const fb::DeviceResources res{0};
    dev = res.device();
    d.emplace(kCount, res.device());
  }

  fb::Buffers::host<float> in(kCount);
  fb::Buffers::host<float> out(kCount);
  for (std::size_t i = 0; i < kCount; ++i)
    in.data()[i] = static_cast<float>(i);

  const wwr::wwrStream_t stream = dev->stream();
  // Qualified: a bare wwrStream_t carries no ADL into wwr::extension, so an
  // unqualified sync() would find POSIX ::sync instead.
  ASSERT_EQ(ext::copy(*d, in, stream), wwr::wwrSuccess);
  ASSERT_EQ(ext::copy(out, *d, stream), wwr::wwrSuccess);
  ASSERT_EQ(ext::sync(stream), wwr::wwrSuccess);

  for (std::size_t i = 0; i < kCount; ++i)
    ASSERT_EQ(out.data()[i], static_cast<float>(i)) << "at " << i;
}

// Moving hands over the same Device and handles; the source keeps nothing.
TEST(DeviceResources, MoveTransfersOwnership) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources src{0};
  const fb::Device *device = src.device().get();
  const wwr::wwrStream_t stream = src.stream();

  fb::DeviceResources dst{std::move(src)};

  EXPECT_EQ(dst.device().get(), device);
  EXPECT_EQ(dst.stream(), stream);
  EXPECT_TRUE(dst.blas().valid());
  EXPECT_TRUE(dst.solver().valid());
  EXPECT_EQ(dst.blas().stream(), stream);

  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is the point.
  EXPECT_EQ(src.device(), nullptr);
  EXPECT_FALSE(src.blas().valid());
  EXPECT_FALSE(src.solver().valid());
}

} // namespace
