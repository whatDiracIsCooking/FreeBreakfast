// GPU cases for experimental.fp16_gemm: does this backend's BLAS multiply
// __half matrices, under both accumulators, and to fp16 accuracy?
//
// The inputs are drawn in [-1, 1] and rounded to __half on the host; the
// reference is the double-precision product of those rounded values, so only
// the GEMM's own error is measured. Each entry is held to the worst-case
// recursive-summation bound, gamma_k S + u_half |ref|, where S = sum |a_il b_lj|
// and gamma_k = k u / (1 - k u) in the accumulator's unit roundoff u (the
// products of two halves are exact in either). Both bounds are rigorous, so a
// failure is a wrong answer, not a tight tolerance.
//
// Shapes are odd and even, below and past the 8- and 16-wide tiles tensor
// cores take, so both a vendor's aligned fast path and its edge handling run.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.fp16_gemm;
import wwr.fp16;

namespace {

namespace fb = free_breakfast;

using Half = wwr::wwrHalf;

constexpr double kUnitHalf = 0x1p-11;
constexpr double kUnitFloat = 0x1p-24;

struct Shape {
  std::int32_t m;
  std::int32_t n;
  std::int32_t k;
};

std::vector<Shape> shapes() {
  return {{1, 1, 1},    {7, 5, 3},      {8, 8, 8},       {16, 16, 16},    {33, 17, 9},
          {64, 64, 64}, {127, 129, 65}, {256, 128, 512}, {500, 300, 700}, {1024, 1024, 64}};
}

// Column-major, values in [-1, 1] already rounded to __half.
std::vector<Half> random_half(const std::size_t count, const std::uint64_t seed) {
  std::mt19937_64 gen{seed};
  std::uniform_real_distribution<float> dist{-1.0F, 1.0F};
  std::vector<Half> out(count);
  for (auto &x : out) {
    x = wwr::wwrFloat2Half(dist(gen));
  }
  return out;
}

double to_double(const Half h) {
  return static_cast<double>(wwr::wwrHalf2Float(h));
}

// C = A B on the device; the download is synchronized.
std::vector<Half> device_gemm(fb::DeviceResources &res, const fb::Fp16Accumulate accumulate,
                              const Shape s, const std::vector<Half> &a,
                              const std::vector<Half> &b) {
  const auto mk = static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.k);
  const auto kn = static_cast<std::size_t>(s.k) * static_cast<std::size_t>(s.n);
  const auto mn = static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.n);
  fb::Buffers::device<Half> da{mk, res.device()};
  fb::Buffers::device<Half> db{kn, res.device()};
  fb::Buffers::device<Half> dc{mn, res.device()};
  const wwr::wwrStream_t stream = res.stream();
  EXPECT_EQ(wwr::wwrMemcpyAsync(da.data(), a.data(), mk * sizeof(Half), wwr::wwrMemcpyHostToDevice,
                                stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(db.data(), b.data(), kn * sizeof(Half), wwr::wwrMemcpyHostToDevice,
                                stream),
            wwr::wwrSuccess);
  EXPECT_EQ(fb::gemm_fp16(res, accumulate, s.m, s.n, s.k, da.data(), db.data(), dc.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  std::vector<Half> c(mn);
  EXPECT_EQ(wwr::wwrMemcpyAsync(c.data(), dc.data(), mn * sizeof(Half), wwr::wwrMemcpyDeviceToHost,
                                stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return c;
}

// Every entry of `got` within the accumulator's bound of the exact product,
// and finite. Reports the first failure and the worst error relative to S.
void expect_within_bound(const Shape s, const std::vector<Half> &a, const std::vector<Half> &b,
                         const std::vector<Half> &got, const double unit) {
  const double ku = static_cast<double>(s.k) * unit;
  ASSERT_LT(ku, 1.0);
  const double gamma = ku / (1.0 - ku);
  double worst = 0.0;
  for (std::int32_t j = 0; j < s.n; ++j) {
    for (std::int32_t i = 0; i < s.m; ++i) {
      double ref = 0.0;
      double abs_sum = 0.0;
      for (std::int32_t l = 0; l < s.k; ++l) {
        const double p = to_double(a[static_cast<std::size_t>(l * s.m + i)]) *
                         to_double(b[static_cast<std::size_t>(j * s.k + l)]);
        ref += p;
        abs_sum += std::abs(p);
      }
      const double c = to_double(got[static_cast<std::size_t>(j * s.m + i)]);
      const double err = std::abs(c - ref);
      const double bound = gamma * abs_sum + kUnitHalf * std::abs(ref);
      ASSERT_TRUE(std::isfinite(c)) << "C(" << i << ", " << j << ") = " << c;
      ASSERT_LE(err, bound) << "C(" << i << ", " << j << ") = " << c << ", exact " << ref;
      if (abs_sum > 0.0) {
        worst = std::max(worst, err / abs_sum);
      }
    }
  }
  ::testing::Test::RecordProperty("worst_err_over_abs_sum", std::format("{:.3e}", worst));
}

void check_shapes(const fb::Fp16Accumulate accumulate, const double unit) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res;
  for (const Shape s : shapes()) {
    SCOPED_TRACE(std::format("m = {}, n = {}, k = {}", s.m, s.n, s.k));
    const auto a = random_half(static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.k),
                               11U + static_cast<std::uint64_t>(s.m));
    const auto b = random_half(static_cast<std::size_t>(s.k) * static_cast<std::size_t>(s.n),
                               97U + static_cast<std::uint64_t>(s.n));
    expect_within_bound(s, a, b, device_gemm(res, accumulate, s, a, b), unit);
  }
}

TEST(Fp16GemmGpu, HalfAccumulateMatchesExactProduct) {
  check_shapes(fb::Fp16Accumulate::kHalf, kUnitHalf);
}

TEST(Fp16GemmGpu, FloatAccumulateMatchesExactProduct) {
  check_shapes(fb::Fp16Accumulate::kFloat, kUnitFloat);
}

} // namespace
