// GPU cases for experimental.sort.merge: merge_sort of doubles against a host
// sort under std::strong_order -- IEEE-754 total order, the order merge_sort
// promises -- compared bit-for-bit, and against radix_sort, which promises the
// same order and so must agree on every key bit and every carried index.
//
// Sizes straddle the edges of merge_sort's tile (kTile = 256 threads * 11 keys
// = 2816 in merge_sort.cu): within one tile, one tile plus one (a single merge
// pass with a one-key right run), an odd tile count (a lone run carried
// through a pass), and many passes. kTile is restated here, not imported: the
// tests see only the module, and the constant lives in the device code.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.sort.merge;
import free_breakfast.experimental.sort.radix;

namespace {

namespace fb = free_breakfast;

using Keys = std::vector<double>;
using Values = std::vector<std::uint32_t>;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int32_t kTile = 2816;
// The last double in total order (+NaN, every payload bit set), whose ordered
// bits are all ones: the key merge_sort pads a partial tile with. A real one
// must still come out, and ahead of the padding.
const double kHighestNaN = std::bit_cast<double>(std::uint64_t{0x7fff'ffff'ffff'ffff});
// The first double in total order: -NaN, every payload bit set.
const double kLowestNaN = std::bit_cast<double>(~std::uint64_t{0});

constexpr auto kTotalLess = [](const double a, const double b) {
  return std::strong_order(a, b) < 0;
};

// A lone tile runs in the fewest warps that hold it, so 352, 704 and 1408 keys
// (1, 2 and 4 warps of 11 keys a thread, on 32-lane warps) are edges too.
std::vector<std::int32_t> sizes() {
  return {1,
          2,
          3,
          11,
          12,
          255,
          256,
          352,
          353,
          705,
          1408,
          1409,
          kTile - 1,
          kTile,
          kTile + 1,
          2 * kTile,
          3 * kTile,
          3 * kTile + 1,
          5 * kTile + 7,
          16'384,
          100'003,
          1 << 20};
}

Keys random_bit_keys(const std::size_t n, const std::uint64_t seed) {
  std::mt19937_64 gen{seed};
  Keys keys(n);
  for (auto &k : keys) {
    k = std::bit_cast<double>(gen());
  }
  return keys;
}

// A sort, as a callable: merge_sort and radix_sort take a defaulted argument,
// which a function pointer would drop.
constexpr auto kMergeSort = [](fb::DeviceResources &res, const std::int32_t n, double *k,
                               std::uint32_t *v) { return fb::merge_sort(res, n, k, v); };
constexpr auto kRadixSort = [](fb::DeviceResources &res, const std::int32_t n, double *k,
                               std::uint32_t *v) { return fb::radix_sort(res, n, k, v); };
constexpr auto kVendorMergeSort = [](fb::DeviceResources &res, const std::int32_t n, double *k,
                                     std::uint32_t *v) { return fb::cub_merge_sort(res, n, k, v); };

// Sorts on the device, staging through device buffers.
struct DeviceSort {
  fb::DeviceResources &res;
  std::int32_t n;
  std::size_t count = static_cast<std::size_t>(n);
  fb::Buffers::device<double> keys{count, res.device()};
  fb::Buffers::device<std::uint32_t> values{count, res.device()};

  template <typename T>
  void upload(fb::Buffers::device<T> &d, const std::vector<T> &host) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), count * sizeof(T),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
  }

  template <typename T>
  std::vector<T> download(const fb::Buffers::device<T> &d) {
    std::vector<T> host(count);
    EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), count * sizeof(T),
                                  wwr::wwrMemcpyDeviceToHost, res.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
    return host;
  }

  Keys sort_keys(const Keys &host) {
    upload(keys, host);
    EXPECT_TRUE(fb::merge_sort(res, n, keys.data()));
    return download(keys);
  }

  // Keys, and their input indices as values, through either sort.
  template <typename Sort>
  std::pair<Keys, Values> sort_pairs(const Sort sort, const Keys &host) {
    Values index(count);
    std::iota(index.begin(), index.end(), 0U);
    upload(keys, host);
    upload(values, index);
    EXPECT_TRUE(sort(res, n, keys.data(), values.data()));
    Keys sorted_keys = download(keys);
    return {std::move(sorted_keys), download(values)};
  }
};

void expect_same_bits(const Keys &got, const Keys &want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(got[i]) != std::bit_cast<std::uint64_t>(want[i])) {
      ADD_FAILURE() << std::format("index {}: got {} ({:#018x}), expected {} ({:#018x})", i, got[i],
                                   std::bit_cast<std::uint64_t>(got[i]), want[i],
                                   std::bit_cast<std::uint64_t>(want[i]));
      return;
    }
  }
}

void expect_same(const Values &got, const Values &want) {
  ASSERT_EQ(got.size(), want.size());
  const auto [g, w] = std::ranges::mismatch(got, want);
  if (g != got.end()) {
    ADD_FAILURE() << "index " << (g - got.begin()) << ": got " << *g << ", expected " << *w;
  }
}

TEST(MergeSortGpu, MatchesTotalOrderOnRandomBits) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    const Keys in = random_bit_keys(static_cast<std::size_t>(n), 5678U + n);
    Keys want = in;
    std::ranges::sort(want, kTotalLess);
    DeviceSort sort{res, n};
    expect_same_bits(sort.sort_keys(in), want);
  }
}

TEST(MergeSortGpu, MatchesStdSortOnFiniteValues) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::int32_t n = 50'000;
  std::mt19937_64 gen{78};
  std::uniform_real_distribution<double> mantissa{-1.0, 1.0};
  std::uniform_int_distribution<std::int32_t> exponent{-20, 20};
  Keys in(static_cast<std::size_t>(n));
  for (auto &k : in) {
    k = std::ldexp(mantissa(gen), exponent(gen));
  }
  Keys want = in;
  std::ranges::sort(want);
  DeviceSort sort{res, n};
  expect_same_bits(sort.sort_keys(in), want);
}

// Every special value, shuffled, against the order written out by hand: the
// NaNs of largest payload at either end, then radix_sort's list.
TEST(MergeSortGpu, SpecialValuesInTotalOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  constexpr double kMax = std::numeric_limits<double>::max();
  constexpr double kMinNormal = std::numeric_limits<double>::min();
  constexpr double kMinSubnormal = std::numeric_limits<double>::denorm_min();
  const Keys want{kLowestNaN,     -kNaN, -kInf, -kMax,         -1.0,       -kMinNormal,
                  -kMinSubnormal, -0.0,  0.0,   kMinSubnormal, kMinNormal, 1.0,
                  kMax,           kInf,  kNaN,  kHighestNaN};
  ASSERT_TRUE(std::signbit(want[1]));
  Keys in = want;
  std::ranges::shuffle(in, std::mt19937{6});
  DeviceSort sort{res, static_cast<std::int32_t>(in.size())};
  expect_same_bits(sort.sort_keys(in), want);
}

// Already sorted, reversed, and all equal.
TEST(MergeSortGpu, OrderedInputs) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::int32_t n = 20'000;
  Keys ascending(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < ascending.size(); ++i) {
    ascending[i] = -1000.25 + 0.5 * static_cast<double>(i);
  }
  const Keys descending(ascending.rbegin(), ascending.rend());
  const Keys equal(static_cast<std::size_t>(n), 3.25);
  DeviceSort sort{res, n};
  expect_same_bits(sort.sort_keys(ascending), ascending);
  expect_same_bits(sort.sort_keys(descending), ascending);
  expect_same_bits(sort.sort_keys(equal), equal);
}

// Few distinct keys -- -0.0 and +0.0 among them, and kHighestNaN, the pad --
// each repeated across many tiles, with the input index as the value: the
// values must be exactly std::stable_sort's permutation.
TEST(MergeSortGpu, CarriesValuesStably) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::array<double, 8> kDistinct{-kInf, -2.5, -0.0, 0.0, 1e-300, 7.0, kNaN, kHighestNaN};
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    const auto count = static_cast<std::size_t>(n);
    std::mt19937 gen{98};
    std::uniform_int_distribution<std::size_t> pick{0, kDistinct.size() - 1};
    Keys keys(count);
    for (auto &k : keys) {
      k = kDistinct[pick(gen)];
    }
    Values index(count);
    std::iota(index.begin(), index.end(), 0U);

    Values want_index = index;
    std::ranges::stable_sort(want_index, kTotalLess,
                             [&](const std::uint32_t i) { return keys[i]; });
    Keys want_keys(count);
    for (std::size_t i = 0; i < count; ++i) {
      want_keys[i] = keys[want_index[i]];
    }

    DeviceSort sort{res, n};
    const auto [got_keys, got_index] = sort.sort_pairs(kMergeSort, keys);
    expect_same_bits(got_keys, want_keys);
    expect_same(got_index, want_index);
  }
}

TEST(MergeSortGpu, EmptyIsANoOp) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  EXPECT_TRUE(fb::merge_sort(res, 0, nullptr));
  EXPECT_TRUE(fb::merge_sort(res, -3, nullptr, nullptr));
  EXPECT_TRUE(fb::cub_merge_sort(res, 0, nullptr));
}

// The same order, so the same keys and the same stable permutation, zeros
// included: radix_sort, and the vendor's merge sort given a total-order
// comparator.
TEST(MergeSortGpu, AgreesWithRadixAndVendorMergeSorts) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    Keys in = random_bit_keys(static_cast<std::size_t>(n), 8765U + n);
    for (std::size_t i = 0; i < in.size(); i += 7) {
      in[i] = i % 2 == 0 ? 0.0 : -0.0;
    }
    DeviceSort sort{res, n};
    const auto [merge_keys, merge_index] = sort.sort_pairs(kMergeSort, in);
    const auto [radix_keys, radix_index] = sort.sort_pairs(kRadixSort, in);
    const auto [vendor_keys, vendor_index] = sort.sort_pairs(kVendorMergeSort, in);
    expect_same_bits(merge_keys, radix_keys);
    expect_same(merge_index, radix_index);
    expect_same_bits(merge_keys, vendor_keys);
    expect_same(merge_index, vendor_index);
  }
}

} // namespace
