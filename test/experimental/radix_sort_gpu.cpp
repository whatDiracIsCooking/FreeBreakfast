// GPU cases for experimental.sort.radix: radix_sort of doubles against a host
// sort under std::strong_order -- IEEE-754 total order, the order radix_sort
// promises -- compared bit-for-bit, so -0.0 against +0.0 and every NaN
// payload are checked exactly rather than through ==.
//
// Sizes straddle the tile edges (one key, a tile, a tile plus one, many
// tiles). The key sets are random bit patterns (every double, NaNs and
// subnormals included), ordinary finite values, the special values in a fixed
// expected order, and duplicate-heavy keys sorted with their input indices as
// values, which checks stability exactly: the values must come out as
// std::stable_sort's permutation.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.sort.radix;

namespace {

namespace fb = free_breakfast;

using Keys = std::vector<double>;
using Values = std::vector<std::uint32_t>;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The order radix_sort sorts in.
constexpr auto kTotalLess = [](const double a, const double b) {
  return std::strong_order(a, b) < 0;
};

// Both paths, and the edges of each. Up to 16384 keys: one tile of up to 1024
// keys in the fewest warps that hold it (1 to 128 to 1024 keys), then tiles
// merged in rounds -- 1025 and 2049 are one and two merge rounds with a short
// last tile, 4097, 8193 and 16383 three to four, 16384 sixteen full tiles.
// Past it, the multi-kernel path's tile is kFourWarps keys, and 16385 leaves
// it a one-key last tile.
std::vector<std::int32_t> sizes() {
  const auto tile = static_cast<std::int32_t>(fb::kFourWarps);
  return {1,    2,    31,   tile - 1, tile,   tile + 1, 3 * tile + 7, 1024,           1025,
          2049, 4097, 8193, 10'000,   16'383, 16'384,   16'385,       3 * 16'384 + 5, 1 << 20};
}

// Uniform over bit patterns, so over every kind of double: about 1 in 2000 is
// a NaN or an infinity, and both signs and the whole exponent range appear.
Keys random_bit_keys(const std::size_t n, const std::uint64_t seed) {
  std::mt19937_64 gen{seed};
  Keys keys(n);
  for (auto &k : keys) {
    k = std::bit_cast<double>(gen());
  }
  return keys;
}

// Sorts on the device through radix_sort, staging through device buffers.
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
    EXPECT_TRUE(fb::radix_sort(res, n, keys.data()));
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

// The two sorts as callables: radix_sort and cub_radix_sort are functions with
// a defaulted argument, which a function pointer would drop.
constexpr auto kOurSort = [](fb::DeviceResources &res, const std::int32_t n, double *k,
                             std::uint32_t *v) { return fb::radix_sort(res, n, k, v); };
constexpr auto kVendorSort = [](fb::DeviceResources &res, const std::int32_t n, double *k,
                                std::uint32_t *v) { return fb::cub_radix_sort(res, n, k, v); };

// Bit-for-bit, reporting the first disagreement only: a million-element diff
// is noise.
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

TEST(RadixSortGpu, MatchesTotalOrderOnRandomBits) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    const Keys in = random_bit_keys(static_cast<std::size_t>(n), 1234U + n);
    Keys want = in;
    std::ranges::sort(want, kTotalLess);
    DeviceSort sort{res, n};
    expect_same_bits(sort.sort_keys(in), want);
  }
}

// What the sort will mostly see: finite values of both signs over a few
// decades, where total order and operator< agree.
TEST(RadixSortGpu, MatchesStdSortOnFiniteValues) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::int32_t n = 50'000;
  std::mt19937_64 gen{77};
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

// Every special value, shuffled, against the order written out by hand.
TEST(RadixSortGpu, SpecialValuesInTotalOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  constexpr double kMax = std::numeric_limits<double>::max();
  constexpr double kMinNormal = std::numeric_limits<double>::min();
  constexpr double kMinSubnormal = std::numeric_limits<double>::denorm_min();
  const Keys want{-kNaN, -kInf,         -kMax,      -1.0, -kMinNormal, -kMinSubnormal, -0.0,
                  0.0,   kMinSubnormal, kMinNormal, 1.0,  kMax,        kInf,           kNaN};
  // -kNaN must really carry the sign bit, or the expected order is wrong.
  ASSERT_TRUE(std::signbit(want.front()));
  Keys in = want;
  std::ranges::shuffle(in, std::mt19937{5});
  DeviceSort sort{res, static_cast<std::int32_t>(in.size())};
  expect_same_bits(sort.sort_keys(in), want);
}

// Already sorted, reversed, and all equal.
TEST(RadixSortGpu, OrderedInputs) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::int32_t n = 5000;
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

// Few distinct keys -- -0.0 and +0.0 among them, which total order separates
// -- each repeated across many tiles, with the input index as the value: the
// values must be exactly std::stable_sort's permutation.
TEST(RadixSortGpu, CarriesValuesStably) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const std::array<double, 7> kDistinct{-kInf, -2.5, -0.0, 0.0, 1e-300, 7.0, kNaN};
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    const auto count = static_cast<std::size_t>(n);
    std::mt19937 gen{99};
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
    sort.upload(sort.keys, keys);
    sort.upload(sort.values, index);
    ASSERT_TRUE(fb::radix_sort(res, n, sort.keys.data(), sort.values.data()));
    expect_same_bits(sort.download(sort.keys), want_keys);
    expect_same(sort.download(sort.values), want_index);
  }
}

// n <= 0 touches nothing and succeeds; null keys prove nothing is read.
TEST(RadixSortGpu, EmptyIsANoOp) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  EXPECT_TRUE(fb::radix_sort(res, 0, nullptr));
  EXPECT_TRUE(fb::radix_sort(res, -3, nullptr, nullptr));
  EXPECT_TRUE(fb::cub_radix_sort(res, 0, nullptr));
}

// Away from zero the vendor's order is total order too, NaNs included, and it
// is stable, so on random bit patterns -- where a zero practically never
// appears -- the two sorts must agree on every key bit and every index.
TEST(RadixSortGpu, AgreesWithVendorSort) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  for (const std::int32_t n : sizes()) {
    SCOPED_TRACE(std::format("n = {}", n));
    const Keys in = random_bit_keys(static_cast<std::size_t>(n), 4321U + n);
    DeviceSort sort{res, n};
    const auto [our_keys, our_index] = sort.sort_pairs(kOurSort, in);
    const auto [vendor_keys, vendor_index] = sort.sort_pairs(kVendorSort, in);
    expect_same_bits(our_keys, vendor_keys);
    expect_same(our_index, vendor_index);
  }
}

// The one place the orders part: radix_sort puts every -0.0 before every
// +0.0, while the vendor sorts the two as one key, so mixed zeros keep their
// input order. Pinned so the README's claim is checked, not remembered.
TEST(RadixSortGpu, VendorTreatsSignedZerosAsEqual) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  const Keys in{0.0, -0.0, 1.0, -1.0, 0.0, -0.0};
  DeviceSort sort{res, static_cast<std::int32_t>(in.size())};

  const auto [our_keys, our_index] = sort.sort_pairs(kOurSort, in);
  expect_same_bits(our_keys, Keys{-1.0, -0.0, -0.0, 0.0, 0.0, 1.0});
  expect_same(our_index, Values{3, 1, 5, 0, 4, 2});

  const auto [vendor_keys, vendor_index] = sort.sort_pairs(kVendorSort, in);
  expect_same_bits(vendor_keys, Keys{-1.0, 0.0, -0.0, 0.0, -0.0, 1.0});
  expect_same(vendor_index, Values{3, 0, 1, 4, 5, 2});
}

} // namespace
