// GPU cases for recipes.sort, in the default build: merge_sort's keys and
// permutation against std::stable_sort, within one tile and past it (kTile =
// 2816 in merge_sort.cu), with ties so stability shows; and gather_columns
// against a host gather of a merge_sort permutation, plus its argument checks.
// The sort's full suite -- total order, NaNs, zeros, agreement with radix_sort
// and the vendor sort -- is test/experimental/merge_sort_gpu.cpp's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes.sort;

namespace {

namespace fb = free_breakfast;

template <typename T>
fb::Buffers::device<T> upload(fb::DeviceResources &res, const std::vector<T> &host) {
  fb::Buffers::device<T> buffer(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(buffer.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return buffer;
}

template <typename T>
std::vector<T> download(fb::DeviceResources &res, const fb::Buffers::device<T> &buffer) {
  std::vector<T> host(buffer.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), buffer.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// n keys drawn from 0.25 * [-k, k] for a small k, so most of them tie.
std::vector<double> tied_keys(const std::size_t n, const std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_int_distribution<std::int32_t> dist{-40, 40};
  std::vector<double> keys(n);
  for (double &key : keys) {
    key = 0.25 * dist(gen);
  }
  return keys;
}

class MergeSortSizes : public ::testing::TestWithParam<std::size_t> {};

TEST_P(MergeSortSizes, MatchesStableSort) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const std::size_t n = GetParam();
  const std::vector<double> keys = tied_keys(n, static_cast<std::uint32_t>(n));
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), std::uint32_t{0});
  auto d_keys = upload(res, keys);
  auto d_perm = upload(res, order);
  ASSERT_TRUE(fb::merge_sort(res, static_cast<std::int32_t>(n), d_keys.data(), d_perm.data()));

  std::ranges::stable_sort(order, {}, [&](const std::uint32_t i) { return keys[i]; });
  const std::vector<std::uint32_t> perm = download(res, d_perm);
  const std::vector<double> sorted = download(res, d_keys);
  EXPECT_EQ(perm, order);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(sorted[i], keys[order[i]]) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(AroundOneTile, MergeSortSizes, ::testing::Values(1, 37, 2816, 2817, 9000));

TEST(GatherColumns, AppliesASortsPermutation) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  constexpr std::int32_t kRows = 41;
  constexpr std::int32_t kCols = 13;
  const std::vector<double> keys = tied_keys(kCols, 7);
  std::vector<std::uint32_t> order(kCols);
  std::iota(order.begin(), order.end(), std::uint32_t{0});
  std::vector<double> src(static_cast<std::size_t>(kRows) * kCols);
  std::iota(src.begin(), src.end(), 0.0);
  auto d_keys = upload(res, keys);
  auto d_perm = upload(res, order);
  auto d_src = upload(res, src);
  fb::Buffers::device<double> d_dst(src.size(), res.device());
  ASSERT_TRUE(fb::merge_sort(res, kCols, d_keys.data(), d_perm.data()));
  ASSERT_TRUE(fb::gather_columns(res, kRows, kCols, d_perm.data(), d_src.data(), d_dst.data()));

  const std::vector<std::uint32_t> perm = download(res, d_perm);
  const std::vector<double> dst = download(res, d_dst);
  for (std::int32_t j = 0; j < kCols; ++j) {
    for (std::int32_t r = 0; r < kRows; ++r) {
      EXPECT_EQ(dst[r + (static_cast<std::size_t>(kRows) * j)],
                src[r + (static_cast<std::size_t>(kRows) * perm[j])])
          << "row " << r << ", column " << j;
    }
  }
}

TEST(GatherColumns, RejectsBadArguments) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  EXPECT_TRUE(fb::gather_columns(res, 0, 5, nullptr, nullptr, nullptr));
  EXPECT_TRUE(fb::gather_columns(res, 5, 0, nullptr, nullptr, nullptr));
  for (const auto &[rows, cols] : {std::pair{-1, 5}, std::pair{5, -1}, std::pair{5, 5}}) {
    const fb::Status status = fb::gather_columns(res, rows, cols, nullptr, nullptr, nullptr);
    ASSERT_FALSE(status.has_value()) << rows << " x " << cols;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
