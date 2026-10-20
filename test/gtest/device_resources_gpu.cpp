// GPU smoke cases for resources:device_resources: DeviceResources constructed on a
// real device, its handles bound to the Device's stream, the one-way
// ownership that lets a buffer outlive the bundle that handed out its Device,
// and the solver scratch: grow-only workspaces and the info flag.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::size_t kCount = 1024;

// Device is this tree's own pool-tier handle: the full wwr ladder, so Buffers,
// BlasHandle and SolverHandle can bind to it.
static_assert(ext::device_handle_pool<fb::Device>);
static_assert(std::is_move_constructible_v<fb::Device>);
static_assert(!std::is_copy_constructible_v<fb::Device>);

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

// The work stream every handle is bound to is the Device's ForkStream's.
TEST(DeviceResources, WorkStreamIsTheForkStream) {
  FBF_REQUIRE_GPU();

  const fb::DeviceResources res{0};

  EXPECT_EQ(res.fork_stream().dev_idx(), res.dev_idx());
  EXPECT_EQ(static_cast<wwr::wwrStream_t>(res.fork_stream().stream()), res.stream());
  EXPECT_EQ(&res.fork_stream(), &res.device()->fork_stream());
  EXPECT_GT(res.device()->props().totalGlobalMem, 0U);
}

// The whole working set joins another stream's capture through its ForkStream:
// a copy queued on res.stream() between fork_from and join_into lands in the
// parent's graph, and runs only when that graph is launched. The destructors
// run inside the inner scope so the final check sees any error they leave.
TEST(DeviceResources, ForksIntoAParentCapture) {
  FBF_REQUIRE_GPU();

  {
    fb::Stream parent{0};
    fb::DeviceResources res{0};

    fb::Buffers::device<float> d(kCount, res.device());
    fb::Buffers::pinned<float> in(kCount);
    fb::Buffers::pinned<float> out(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
      in.data()[i] = static_cast<float>(3 * i);
      out.data()[i] = -1.0F;
    }
    ASSERT_EQ(ext::sync(res.stream()), wwr::wwrSuccess); // the allocation, outside capture

    fb::ForkStream &fork = res.fork_stream();
    ASSERT_EQ(ext::begin_capture(parent), wwr::wwrSuccess);
    ASSERT_EQ(fork.fork_from(parent), wwr::wwrSuccess);
    ASSERT_EQ(ext::copy(d, in, res.stream()), wwr::wwrSuccess);
    ASSERT_EQ(ext::copy(out, d, res.stream()), wwr::wwrSuccess);
    ASSERT_EQ(fork.record_join(), wwr::wwrSuccess);
    ASSERT_EQ(fork.join_into(parent), wwr::wwrSuccess);
    auto graph = parent.end_capture();
    ASSERT_TRUE(graph.valid());

    ASSERT_EQ(ext::sync(res.stream()), wwr::wwrSuccess);
    EXPECT_EQ(out.data()[0], -1.0F); // captured, not run

    auto exec = graph.instantiate();
    ASSERT_TRUE(exec.valid());
    ASSERT_EQ(ext::launch(exec, parent), wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(parent), wwr::wwrSuccess);

    for (std::size_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(out.data()[i], static_cast<float>(3 * i)) << "at " << i;
    }
  }

  EXPECT_EQ(wwr::wwrGetLastError(), wwr::wwrSuccess);
}

// The Device's pool caches up to the requested release threshold: 1 GiB by
// default, overriding wwr's 1 MiB, or whatever the caller passes.
TEST(DeviceResources, PoolReleaseThreshold) {
  FBF_REQUIRE_GPU();

  const auto threshold = [](const fb::DeviceResources &res) {
    std::uint64_t value = 0;
    EXPECT_EQ(wwr::wwrMemPoolGetAttribute(res.device()->pool(), wwr::wwrMemPoolAttrReleaseThreshold,
                                          &value),
              wwr::wwrSuccess);
    return value;
  };

  EXPECT_EQ(threshold(fb::DeviceResources{0}), std::uint64_t{1} << 30);
  EXPECT_EQ(fb::DeviceResources::kDefaultPoolReleaseThreshold, std::uint64_t{1} << 30);

  constexpr std::uint64_t kCustom = std::uint64_t{256} << 20;
  EXPECT_EQ(threshold(fb::DeviceResources{0, kCustom}), kCustom);
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

// Both workspaces start empty, grow in kWorkspaceAlignment steps, and never
// shrink: a smaller request hands back the same block.
TEST(DeviceResources, WorkspacesGrowAlignedAndNeverShrink) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  fb::DeviceResources res{0};

  EXPECT_EQ(res.device_workspace(0).num_elements(), 0U);
  EXPECT_EQ(res.device_workspace(1).num_elements(), kAlign);
  EXPECT_EQ(res.device_workspace(kAlign + 1).num_elements(), 2 * kAlign);
  const std::byte *grown = res.device_workspace(0).data();
  EXPECT_EQ(res.device_workspace(kAlign).data(), grown);
  EXPECT_EQ(res.device_workspace(kAlign).num_elements(), 2 * kAlign);

  EXPECT_EQ(res.host_workspace(0).num_elements(), 0U);
  EXPECT_EQ(res.host_workspace(3 * kAlign - 1).num_elements(), 3 * kAlign);
  const std::byte *host = res.host_workspace(0).data();
  EXPECT_EQ(res.host_workspace(1).data(), host);
}

// The rounding is what lets one byte workspace serve every solver element type:
// a request that is no multiple of 16 still views whole as complex<double>.
TEST(DeviceResources, WorkspaceViewsAsAnyElementType) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  auto &ws = res.device_workspace(1004);

  EXPECT_EQ(ext::reinterpret_buffer_view<std::complex<double>>(ws).num_elements(),
            ws.num_elements() / sizeof(std::complex<double>));
  EXPECT_EQ(ext::reinterpret_buffer_view<float>(ws).num_elements(),
            ws.num_elements() / sizeof(float));
  EXPECT_GE(ext::reinterpret_buffer_view<double>(ws).num_elements() * sizeof(double), 1004U);
}

// fetch_info() reads back what was last written to info() on the work stream:
// zero from the allocation, then whatever a later write leaves.
TEST(DeviceResources, FetchInfoReadsTheDeviceFlag) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  ASSERT_EQ(res.info().num_elements(), 1U);
  EXPECT_EQ(res.fetch_info().value(), 0);

  const std::int32_t flag = -3;
  ASSERT_EQ(wwr::wwrMemcpyAsync(res.info().data(), &flag, sizeof(flag), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(res.fetch_info().value(), -3);
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
