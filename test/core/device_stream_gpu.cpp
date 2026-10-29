// GPU cases for core/wwr's :device_stream: a DeviceStream made from a Device,
// buffers bound to it drawing from the Device's pool, the Device kept alive by
// its streams, and work handed back to the Device's own stream by fork/join.
// The static_asserts at the top are checked whenever this binary builds,
// device or not.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

// Owns a ForkStream: movable, not copyable.
static_assert(std::is_move_constructible_v<fb::DeviceStream>);
static_assert(std::is_move_assignable_v<fb::DeviceStream>);
static_assert(!std::is_copy_constructible_v<fb::DeviceStream>);
static_assert(!std::is_copy_assignable_v<fb::DeviceStream>);

// dev_idx() + stream() + pool(): the pool tier, as Device is, so StreamBuffers
// allocate from the Device's pool rather than the device's default one.
static_assert(ext::device_handle_pool<fb::DeviceStream>);

constexpr std::size_t kCount = 1024;

TEST(DeviceStream, OwnsAStreamAndSharesTheDevicesPool) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  fb::DeviceStream a{dev};
  fb::DeviceStream b{dev};

  EXPECT_EQ(a.dev_idx(), dev->dev_idx());
  EXPECT_EQ(a.device(), dev);
  EXPECT_TRUE(a.stream().valid());
  EXPECT_NE(a.stream().get(), dev->stream().get());
  EXPECT_NE(a.stream().get(), b.stream().get());
  EXPECT_EQ(&a.stream(), &a.fork_stream().stream());
  EXPECT_EQ(&a.pool(), &dev->pool());
  EXPECT_EQ(&b.pool(), &dev->pool());
}

// The last outside reference to the Device dropped: the DeviceStream's own keeps
// its pool alive for allocation and for the stream-ordered free afterwards.
TEST(DeviceStream, KeepsItsDeviceAlive) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  std::weak_ptr<fb::Device> watch = dev;
  auto ds = std::make_shared<fb::DeviceStream>(std::move(dev));
  ASSERT_FALSE(watch.expired());

  {
    fb::StreamBuffers::device<float> d(kCount, ds);
    fb::StreamBuffers::pinned<float> in(kCount);
    fb::StreamBuffers::pinned<float> out(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
      in.data()[i] = static_cast<float>(i);
    }
    ASSERT_EQ(ext::copy(d, in, ds->stream()), wwr::wwrSuccess);
    ASSERT_EQ(ext::copy(out, d, ds->stream()), wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(ds->stream()), wwr::wwrSuccess);
    for (std::size_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(out.data()[i], static_cast<float>(i)) << "at " << i;
    }
  }

  ds.reset();
  EXPECT_TRUE(watch.expired());
  EXPECT_EQ(wwr::wwrGetLastError(), wwr::wwrSuccess);
}

// Write a buffer on a DeviceStream, join back into the Device's own stream, and
// read it there: the read sees the write only if join_into ordered it.
TEST(DeviceStream, ForkJoinsWithTheDevicesStream) {
  FBF_REQUIRE_GPU();

  auto dev = std::make_shared<fb::Device>(0);
  auto ds = std::make_shared<fb::DeviceStream>(dev);
  fb::Stream &parent = dev->stream();

  fb::StreamBuffers::device<float> d(kCount, ds);
  fb::Buffers::pinned<float> in(kCount);
  fb::Buffers::pinned<float> out(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    in.data()[i] = static_cast<float>(3 * i);
    out.data()[i] = -1.0F;
  }

  ASSERT_EQ(ds->fork_stream().fork_from(parent), wwr::wwrSuccess);
  ASSERT_EQ(ext::copy(d, in, ds->stream()), wwr::wwrSuccess);
  ASSERT_EQ(ds->fork_stream().record_join(), wwr::wwrSuccess);
  ASSERT_EQ(ds->fork_stream().join_into(parent), wwr::wwrSuccess);

  ASSERT_EQ(ext::copy(out, d, parent), wwr::wwrSuccess);
  ASSERT_EQ(ext::sync(parent), wwr::wwrSuccess);

  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(out.data()[i], static_cast<float>(3 * i)) << "at " << i;
  }
}

} // namespace
