// GPU cases for linalg.complex_cast: split_planes and merge_planes on a
// padded block in both precisions, checked bit for bit (both only move
// components), with the padding rows of whichever side is written left as
// sentinels, plus the round trip and an empty block.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.linalg;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kRows = 37; // odd, so no block size divides it
constexpr std::int32_t kCols = 11;
constexpr std::int32_t kLdc = 41; // the complex side's leading dimension
constexpr std::int32_t kLdp = 39; // the planes', != kLdc

template <typename Z>
using Real = wwr::ComplexToRealType<Z>;

// Written over every element before a call, so a padding row the call touched
// no longer holds it.
template <typename R>
constexpr R kSentinel = R{-12345};

template <typename T>
void upload(fb::DeviceResources &res, fb::Buffers::device<T> &d, const std::vector<T> &host) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

template <typename T>
std::vector<T> download(fb::DeviceResources &res, const fb::Buffers::device<T> &d,
                        std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), count * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// A distinct value per (i, j) and component, so a transposed or shifted index
// cannot match. Padding rows hold the sentinel.
template <typename R>
std::vector<R> plane(std::int32_t ld, R offset) {
  std::vector<R> p(static_cast<std::size_t>(ld) * kCols, kSentinel<R>);
  for (std::int32_t j = 0; j < kCols; ++j) {
    for (std::int32_t i = 0; i < kRows; ++i) {
      p[i + (j * ld)] = offset + static_cast<R>(i) + (R{0.25} * static_cast<R>(j));
    }
  }
  return p;
}

template <typename Z>
std::vector<Z> complex_matrix(const std::vector<Real<Z>> &re, const std::vector<Real<Z>> &im) {
  std::vector<Z> z;
  z.reserve(re.size());
  for (std::size_t k = 0; k < re.size(); ++k) {
    z.push_back(Z{re[k], im[k]});
  }
  return z;
}

template <typename Z>
class ComplexCast : public ::testing::Test {};

using ComplexTypes = ::testing::Types<wwr::wwrFloatComplex, wwr::wwrDoubleComplex>;
TYPED_TEST_SUITE(ComplexCast, ComplexTypes);

TYPED_TEST(ComplexCast, SplitWritesEachComponentAndLeavesPaddingAlone) {
  FBF_REQUIRE_GPU();
  using Z = TypeParam;
  using R = Real<Z>;

  fb::DeviceResources res{0};
  const auto a = complex_matrix<Z>(plane<R>(kLdc, R{1}), plane<R>(kLdc, R{-100}));
  const std::vector<R> sentinels(static_cast<std::size_t>(kLdp) * kCols, kSentinel<R>);
  fb::Buffers::device<Z> da(a.size(), res.device());
  fb::Buffers::device<R> dre(sentinels.size(), res.device());
  fb::Buffers::device<R> dim(sentinels.size(), res.device());
  upload(res, da, a);
  upload(res, dre, sentinels);
  upload(res, dim, sentinels);

  ASSERT_TRUE(
      fb::split_planes<Z>(res, kRows, kCols, da.data(), kLdc, dre.data(), dim.data(), kLdp));

  // The expected planes are plane()'s own output at kLdp: padding and all.
  EXPECT_EQ(download(res, dre, sentinels.size()), plane<R>(kLdp, R{1}));
  EXPECT_EQ(download(res, dim, sentinels.size()), plane<R>(kLdp, R{-100}));
}

TYPED_TEST(ComplexCast, MergeBuildsEachElementAndLeavesPaddingAlone) {
  FBF_REQUIRE_GPU();
  using Z = TypeParam;
  using R = Real<Z>;

  fb::DeviceResources res{0};
  const auto re = plane<R>(kLdp, R{1});
  const auto im = plane<R>(kLdp, R{-100});
  const std::vector<Z> sentinels(static_cast<std::size_t>(kLdc) * kCols,
                                 Z{kSentinel<R>, kSentinel<R>});
  fb::Buffers::device<R> dre(re.size(), res.device());
  fb::Buffers::device<R> dim(im.size(), res.device());
  fb::Buffers::device<Z> dc(sentinels.size(), res.device());
  upload(res, dre, re);
  upload(res, dim, im);
  upload(res, dc, sentinels);

  ASSERT_TRUE(
      fb::merge_planes<Z>(res, kRows, kCols, dre.data(), dim.data(), kLdp, dc.data(), kLdc));

  const auto got = download(res, dc, sentinels.size());
  const auto expected = complex_matrix<Z>(plane<R>(kLdc, R{1}), plane<R>(kLdc, R{-100}));
  for (std::size_t k = 0; k < got.size(); ++k) {
    ASSERT_EQ(got[k].x, expected[k].x) << "at " << k;
    ASSERT_EQ(got[k].y, expected[k].y) << "at " << k;
  }
}

// Random components, split then merged into a second matrix: bit-identical.
TYPED_TEST(ComplexCast, SplitThenMergeRoundTripsExactly) {
  FBF_REQUIRE_GPU();
  using Z = TypeParam;
  using R = Real<Z>;

  std::mt19937 gen{2024};
  std::uniform_real_distribution<R> dist{R{-1}, R{1}};
  std::vector<Z> a(static_cast<std::size_t>(kRows) * kCols);
  for (auto &z : a) {
    z = Z{dist(gen), dist(gen)};
  }

  fb::DeviceResources res{0};
  fb::Buffers::device<Z> da(a.size(), res.device());
  fb::Buffers::device<R> dre(a.size(), res.device());
  fb::Buffers::device<R> dim(a.size(), res.device());
  fb::Buffers::device<Z> dc(a.size(), res.device());
  upload(res, da, a);

  ASSERT_TRUE(
      fb::split_planes<Z>(res, kRows, kCols, da.data(), kRows, dre.data(), dim.data(), kRows));
  ASSERT_TRUE(
      fb::merge_planes<Z>(res, kRows, kCols, dre.data(), dim.data(), kRows, dc.data(), kRows));

  const auto got = download(res, dc, a.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    ASSERT_EQ(got[k].x, a[k].x) << "at " << k;
    ASSERT_EQ(got[k].y, a[k].y) << "at " << k;
  }
}

// Null pointers would fault if anything launched.
TYPED_TEST(ComplexCast, EmptyBlockLaunchesNothing) {
  FBF_REQUIRE_GPU();
  using Z = TypeParam;

  fb::DeviceResources res{0};
  EXPECT_TRUE(fb::split_planes<Z>(res, 0, kCols, nullptr, 1, nullptr, nullptr, 1));
  EXPECT_TRUE(fb::merge_planes<Z>(res, kRows, 0, nullptr, nullptr, 1, nullptr, 1));
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

} // namespace
