// GPU cases for recipes.symmetrize: the average with the transpose and both
// triangle mirrors, at orders around the 32-wide tile's edges and in a padded
// block, all checked bit for bit -- the average against the same sum and halving
// on the host, the mirrors as pure copies -- with the padding rows left as
// sentinels; plus the argument checks.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kPad = 3; // lda = n + kPad, so padding rows exist
constexpr double kSentinel = -7.25;

bool same_bits(const double x, const double y) {
  return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
}

// An lda x n block, column-major: entries in [-1, 1] in its first n rows, the
// sentinel in the padding.
std::vector<double> random_block(const std::int32_t n, const std::int32_t lda) {
  std::mt19937 gen{2468};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> a(static_cast<std::size_t>(lda) * static_cast<std::size_t>(n), kSentinel);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < n; ++i) {
      a[i + (static_cast<std::size_t>(lda) * j)] = dist(gen);
    }
  }
  return a;
}

// Upload @p host, apply @p op to the device copy, and return it as it came back.
template <typename Op>
std::vector<double> run(fb::DeviceResources &res, const std::vector<double> &host, Op op) {
  fb::Buffers::device<double> a(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(a.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  const fb::Status status = op(a.data());
  EXPECT_TRUE(status.has_value()) << status.error().message;
  std::vector<double> out(host.size());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), a.data(), out.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

void expect_padding_untouched(const std::vector<double> &got, const std::int32_t n,
                              const std::int32_t lda) {
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = n; i < lda; ++i) {
      ASSERT_TRUE(same_bits(got[i + (static_cast<std::size_t>(lda) * j)], kSentinel))
          << "padding (" << i << ", " << j << ")";
    }
  }
}

class Symmetrize : public ::testing::TestWithParam<std::int32_t> {};

// Every element is the host's (a_ij + a_ji) * 0.5 to the bit, which makes the
// result exactly symmetric.
TEST_P(Symmetrize, AveragesWithTheTransposeBitwise) {
  FBF_REQUIRE_GPU();
  const std::int32_t n = GetParam();
  const std::int32_t lda = n + kPad;
  const auto ld = static_cast<std::size_t>(lda);
  const std::vector<double> a = random_block(n, lda);

  fb::DeviceResources res{0};
  const std::vector<double> got =
      run(res, a, [&](double *d) { return fb::symmetrize(res, n, d, lda); });
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < n; ++i) {
      const double want = (a[i + (ld * j)] + a[j + (ld * i)]) * 0.5;
      ASSERT_TRUE(same_bits(got[i + (ld * j)], want))
          << "(" << i << ", " << j << "): " << got[i + (ld * j)] << " != " << want;
    }
  }
  expect_padding_untouched(got, n, lda);
}

// An already symmetric input comes back unchanged: averaging a pair of equal
// values is exact.
TEST_P(Symmetrize, LeavesASymmetricMatrixUnchanged) {
  FBF_REQUIRE_GPU();
  const std::int32_t n = GetParam();
  const std::int32_t lda = n + kPad;
  const auto ld = static_cast<std::size_t>(lda);
  std::vector<double> a = random_block(n, lda);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < j; ++i) {
      a[i + (ld * j)] = a[j + (ld * i)];
    }
  }

  fb::DeviceResources res{0};
  const std::vector<double> got =
      run(res, a, [&](double *d) { return fb::symmetrize(res, n, d, lda); });
  ASSERT_TRUE(std::ranges::equal(got, a, same_bits));
}

// The upper triangle becomes the lower one's transpose, starting from NaN so a
// read of it would show; the lower triangle, the diagonal and the padding are
// untouched. Then the same from the upper triangle.
TEST_P(Symmetrize, MirrorsEitherTriangleBitwise) {
  FBF_REQUIRE_GPU();
  const std::int32_t n = GetParam();
  const std::int32_t lda = n + kPad;
  const auto ld = static_cast<std::size_t>(lda);
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  fb::DeviceResources res{0};
  for (const bool from_lower : {true, false}) {
    std::vector<double> a = random_block(n, lda);
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        if (from_lower ? i < j : i > j) {
          a[i + (ld * j)] = kNaN;
        }
      }
    }
    const wwr::wwrblasFillMode_t fill =
        from_lower ? wwr::WWRBLAS_FILL_MODE_LOWER : wwr::WWRBLAS_FILL_MODE_UPPER;
    const std::vector<double> got =
        run(res, a, [&](double *d) { return fb::symmetrize(res, fill, n, d, lda); });
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        const bool written = from_lower ? i < j : i > j;
        const double want = written ? a[j + (ld * i)] : a[i + (ld * j)];
        ASSERT_TRUE(same_bits(got[i + (ld * j)], want))
            << (from_lower ? "lower" : "upper") << " (" << i << ", " << j << ")";
      }
    }
    expect_padding_untouched(got, n, lda);
  }
}

// One element, a tile and its edges, two tiles and their edges, and several
// tiles with a ragged last one.
INSTANTIATE_TEST_SUITE_P(Orders, Symmetrize,
                         ::testing::Values(1, 2, 31, 32, 33, 63, 64, 65, 100, 129));

TEST(SymmetrizeArguments, EmptySucceedsAndBadShapesAreInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // Packed and empty, lda = n = 0, as a caller passing its order twice has it.
  EXPECT_TRUE(fb::symmetrize(res, 0, nullptr, 0).has_value());
  EXPECT_TRUE(fb::symmetrize(res, wwr::WWRBLAS_FILL_MODE_LOWER, 0, nullptr, 0).has_value());

  const auto invalid = [](const fb::Status &status) {
    return !status.has_value() && status.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(invalid(fb::symmetrize(res, -1, nullptr, 1)));
  EXPECT_TRUE(invalid(fb::symmetrize(res, 4, nullptr, 3)));
  EXPECT_TRUE(invalid(fb::symmetrize(res, wwr::WWRBLAS_FILL_MODE_UPPER, -1, nullptr, 1)));
  EXPECT_TRUE(invalid(fb::symmetrize(res, wwr::WWRBLAS_FILL_MODE_UPPER, 4, nullptr, 3)));
  // A fill naming neither triangle.
  EXPECT_TRUE(invalid(fb::symmetrize(res, static_cast<wwr::wwrblasFillMode_t>(-1), 4, nullptr, 4)));
}

} // namespace
