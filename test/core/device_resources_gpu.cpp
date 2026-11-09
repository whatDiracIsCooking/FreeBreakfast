// GPU smoke cases for resources:device_resources: DeviceResources constructed on a
// real device, its handles bound to the Device's stream, the one-way
// ownership that lets a buffer outlive the bundle that handed out its Device,
// the solver scratch (grow-only workspaces and the info flag), and the pinned
// memory: the pinned workspace's carve, counted buffers and the pinned cap;
// and the memory budget under a pool cap.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::size_t kCount = 1024;

// Device is this tree's own pool-tier handle: the full wwr ladder, so Buffers,
// BlasHandle and SolverHandle can bind to it.
static_assert(ext::device_handle_pool<fb::Device>);
static_assert(std::is_move_constructible_v<fb::Device>);
static_assert(!std::is_copy_constructible_v<fb::Device>);

// kWorkspaceAlignmentIn<T> is the 256 B step in elements of T, and exists only
// for a T whose size divides it.
template <typename T>
concept HasWorkspaceAlignmentIn = requires { fb::DeviceResources::kWorkspaceAlignmentIn<T>; };
static_assert(fb::DeviceResources::kWorkspaceAlignmentIn<float> == 64);
static_assert(fb::DeviceResources::kWorkspaceAlignmentIn<double> == 32);
static_assert(fb::DeviceResources::kWorkspaceAlignmentIn<std::complex<double>> == 16);
static_assert(!HasWorkspaceAlignmentIn<std::array<char, 24>>);
static_assert(!HasWorkspaceAlignmentIn<std::array<char, 512>>);

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

  EXPECT_EQ(res.pinned_workspace("test", 0).value().size(), 0U);
  EXPECT_EQ(res.pinned_workspace("test", (3 * kAlign) - 1).value().size(), 3 * kAlign);
  const std::byte *pinned = res.pinned_workspace("test", 0).value().data();
  EXPECT_EQ(res.pinned_workspace("test", 1).value().data(), pinned);
  EXPECT_EQ(res.pinned_bytes(), 3 * kAlign);
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

// device_regions lays its regions out in order, each a whole number of
// alignment steps from the workspace's start, in one request for the total;
// a single T is every region's.
TEST(DeviceResources, DeviceRegionsCarveAlignedInOrder) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  fb::DeviceResources res{0};

  const auto [a, b, c] = res.device_regions<double, std::int32_t, std::complex<double>>(
      std::size_t{33}, std::size_t{1}, std::size_t{0});
  static_assert(std::is_same_v<decltype(b), std::int32_t *const>);
  const auto *const base = res.device_workspace(0).data();
  EXPECT_EQ(reinterpret_cast<const std::byte *>(a), base);
  EXPECT_EQ(reinterpret_cast<const std::byte *>(b), base + (2 * kAlign));
  EXPECT_EQ(reinterpret_cast<const std::byte *>(c), base + (3 * kAlign));
  EXPECT_EQ(res.device_workspace(0).num_elements(), 3 * kAlign);

  const auto [x, y] = res.device_regions<float>(std::size_t{64}, std::size_t{65});
  EXPECT_EQ(reinterpret_cast<const std::byte *>(y) - reinterpret_cast<const std::byte *>(x),
            static_cast<std::ptrdiff_t>(kAlign));
  EXPECT_EQ(res.device_workspace(0).num_elements(), 3 * kAlign);
}

// pinned_regions carves the pinned workspace as device_regions carves the
// device one: each region aligned for its type, in order, none overlapping
// the next, in one request for the total.
TEST(DeviceResources, PinnedRegionsCarveAlignedInOrder) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  fb::DeviceResources res{0};

  const auto [a, b, c] = res.pinned_regions<double, std::int32_t, std::complex<double>>(
                                "test", std::size_t{33}, std::size_t{1}, std::size_t{5})
                             .value();
  static_assert(std::is_same_v<decltype(b), std::int32_t *const>);
  const std::byte *const base = res.pinned_workspace("test", 0).value().data();
  EXPECT_EQ(reinterpret_cast<const std::byte *>(a), base);
  EXPECT_EQ(reinterpret_cast<const std::byte *>(b), base + (2 * kAlign));
  EXPECT_EQ(reinterpret_cast<const std::byte *>(c), base + (3 * kAlign));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % alignof(double), 0U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c) % alignof(std::complex<double>), 0U);
  EXPECT_LE(reinterpret_cast<const std::byte *>(a + 33), reinterpret_cast<const std::byte *>(b));
  EXPECT_LE(reinterpret_cast<const std::byte *>(b + 1), reinterpret_cast<const std::byte *>(c));
  EXPECT_EQ(res.pinned_workspace("test", 0).value().size(), 4 * kAlign);

  // Every region is the host's to write, end to end.
  std::fill_n(a, 33, 1.0);
  b[0] = 2;
  std::fill_n(c, 5, std::complex<double>{3.0, 4.0});
  EXPECT_EQ(a[32], 1.0);
  EXPECT_EQ(b[0], 2);
  EXPECT_EQ(c[0], (std::complex<double>{3.0, 4.0}));

  const auto [x, y] = res.pinned_regions<float>("test", std::size_t{64}, std::size_t{65}).value();
  EXPECT_EQ(reinterpret_cast<const std::byte *>(y) - reinterpret_cast<const std::byte *>(x),
            static_cast<std::ptrdiff_t>(kAlign));
  EXPECT_EQ(res.pinned_workspace("test", 0).value().size(), 4 * kAlign);
}

// Under a pinned cap of C, a counted allocation of C/2 fits; a workspace
// growth or a second allocation that would pass C is refused as Unsupported,
// naming the caller and the sizes, and allocates nothing; freeing the counted
// buffer gives its bytes back.
TEST(DeviceResources, PinnedCapRefusesWhatWouldPassIt) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  constexpr std::size_t kCap = 64 * kAlign;
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, kCap};
  EXPECT_EQ(res.pinned_cap(), kCap);
  const std::size_t start = res.pinned_bytes();
  EXPECT_EQ(start, 0U);

  const auto refused = [](const auto &result, const std::string &request) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().kind, fb::ErrorKind::Unsupported);
    const std::string &msg = result.error().message;
    EXPECT_NE(msg.find("caller-x"), std::string::npos) << msg;
    EXPECT_NE(msg.find(request), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string(kCap)), std::string::npos) << msg;
  };

  {
    auto half = res.counted_pinned<double>("caller-x", kCap / 2 / sizeof(double));
    ASSERT_TRUE(half.has_value());
    EXPECT_EQ(half->num_elements(), kCap / 2 / sizeof(double));
    EXPECT_EQ(res.pinned_bytes(), start + (kCap / 2));
    half->data()[0] = 1.0; // the host's to write

    // A growth past what is left: refused, and the workspace is unchanged.
    refused(res.pinned_workspace("caller-x", (kCap / 2) + 1), std::to_string((kCap / 2) + kAlign));
    EXPECT_EQ(res.pinned_workspace("caller-x", 0).value().size(), 0U);
    refused(res.pinned_regions<double>("caller-x", (kCap / 2 / sizeof(double)) + 1),
            std::to_string((kCap / 2) + kAlign));
    EXPECT_EQ(res.pinned_bytes(), start + (kCap / 2));

    // A second allocation past it: refused, and nothing is counted.
    refused(res.counted_pinned<std::byte>("caller-x", (kCap / 2) + 1),
            std::to_string((kCap / 2) + 1));
    EXPECT_EQ(res.pinned_bytes(), start + (kCap / 2));

    // Exactly what is left fits.
    EXPECT_EQ(res.pinned_workspace("caller-x", kCap / 2).value().size(), kCap / 2);
    EXPECT_EQ(res.pinned_bytes(), kCap);
    refused(res.counted_pinned<std::byte>("caller-x", 1), "1");
  }

  // half is freed: only the workspace is left.
  EXPECT_EQ(res.pinned_bytes(), start + (kCap / 2));
  auto again = res.counted_pinned<std::byte>("caller-x", kCap / 2);
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(res.pinned_bytes(), kCap);
}

// Under a pool cap, the budget is the cap less what the pool's live
// allocations use: a live allocation takes exactly its size, rounded to the
// pool's granularity, out of it, and its free gives that back -- the freed
// block stays cached in the pool (a cap is also the release threshold), and
// the budget counts it as available.
TEST(DeviceResources, MemoryBudgetFollowsALiveAllocation) {
  FBF_REQUIRE_GPU();

  constexpr std::uint64_t kCap = std::uint64_t{64} << 20;
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, kCap};
  EXPECT_EQ(res.pool_cap(), kCap);
  const std::size_t granularity = fb::test::pool_granularity(*res.device());
  ASSERT_GT(granularity, 0U);

  const std::size_t start = res.memory_budget().value();
  EXPECT_EQ(start, kCap - res.device()->pool_usage().used_current);

  // Off the granularity, so the rounding shows when it is above 1 B.
  constexpr std::size_t kBytes = (std::size_t{3} << 20) + 8;
  {
    fb::Buffers::device<std::byte> live(kBytes, res.device());
    EXPECT_EQ(start - res.memory_budget().value(), fb::math::align_up(kBytes, granularity));
  }
  ASSERT_TRUE(res.sync().has_value());
  EXPECT_EQ(res.memory_budget().value(), start);
}

// A counted buffer carries its count when moved, and may outlive the
// DeviceResources that allocated it.
TEST(DeviceResources, CountedPinnedMovesItsCount) {
  FBF_REQUIRE_GPU();

  std::optional<fb::CountedPinned<float>> kept;
  {
    fb::DeviceResources res{0};
    auto a = res.counted_pinned<float>("test", kCount).value();
    EXPECT_EQ(res.pinned_bytes(), kCount * sizeof(float));
    fb::CountedPinned<float> b{std::move(a)};
    EXPECT_EQ(res.pinned_bytes(), kCount * sizeof(float));
    b = res.counted_pinned<float>("test", 2 * kCount).value();
    EXPECT_EQ(res.pinned_bytes(), 2 * kCount * sizeof(float));
    kept.emplace(std::move(b));
  }
  ASSERT_EQ(kept->num_elements(), 2 * kCount);
  kept->data()[2 * kCount - 1] = 1.0F;
  kept.reset();
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
