// GPU cases for core/primitives: every free_breakfast::math function evaluated
// on the device -- through primitives.h's device half, in
// test/gpu_kernels/primitives_kernels.cu -- and compared against the host
// standard library on the same inputs.
//
// The reference is std:: rather than the host free_breakfast::math, so a
// mistake shared by both halves of primitives.h (a wrong C name in the
// generator line, say) still shows up. core_vocabulary_host.cpp separately pins
// the host half to std::.
//
// Exact operations must match bit for bit: sqrt is correctly rounded on both
// sides, and the rounding, sign, remainder, min/max and fused functions have
// exactly one right answer. The transcendentals are compared within a few ulps,
// because the vendor device library and the host libm are different
// implementations with their own documented error bounds.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "primitives_kernels.h"

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

namespace fb = free_breakfast;
namespace tk = free_breakfast::test_kernels;

using tk::MathOp;

constexpr std::size_t kCount = 4096;

// Widest gap allowed between the device and host results of a transcendental,
// in units of the result's own epsilon. CUDA documents at most 4 ulps for its
// single-precision functions here (tanf, powf) and 2 for double, and glibc
// adds up to ~1 more on the host side; 8 leaves headroom without hiding a real
// dispatch error, which is off by orders of magnitude.
constexpr double kUlpTolerance = 8.0;

struct Case {
  const char *name;
  MathOp op;
  // Inputs are drawn uniformly from [lo, hi] for x, and the same range for y
  // and z -- every domain below is valid for all three operands it uses.
  double lo;
  double hi;
  bool exact;
};

// What gtest prints for a parameter; without it the log shows a byte dump.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

// Domains keep each function away from its singularities and from overflow in
// float, so a relative comparison is meaningful everywhere.
const std::array kCases{
    Case{"Exp", MathOp::Exp, -20.0, 20.0, false},
    Case{"Exp2", MathOp::Exp2, -30.0, 30.0, false},
    Case{"Expm1", MathOp::Expm1, -5.0, 5.0, false},
    Case{"Log", MathOp::Log, 1e-3, 1e3, false},
    Case{"Log2", MathOp::Log2, 1e-3, 1e3, false},
    Case{"Log10", MathOp::Log10, 1e-3, 1e3, false},
    Case{"Log1p", MathOp::Log1p, -0.9, 100.0, false},
    Case{"Pow", MathOp::Pow, 0.1, 4.0, false},
    Case{"Sqrt", MathOp::Sqrt, 0.0, 1e4, true},
    Case{"Rsqrt", MathOp::Rsqrt, 1e-3, 1e4, false},
    Case{"Sin", MathOp::Sin, -10.0, 10.0, false},
    Case{"Cos", MathOp::Cos, -10.0, 10.0, false},
    Case{"Tan", MathOp::Tan, -1.4, 1.4, false},
    Case{"Asin", MathOp::Asin, -1.0, 1.0, false},
    Case{"Acos", MathOp::Acos, -1.0, 1.0, false},
    Case{"Atan", MathOp::Atan, -50.0, 50.0, false},
    Case{"Atan2", MathOp::Atan2, -5.0, 5.0, false},
    Case{"Sinh", MathOp::Sinh, -10.0, 10.0, false},
    Case{"Cosh", MathOp::Cosh, -10.0, 10.0, false},
    Case{"Tanh", MathOp::Tanh, -5.0, 5.0, false},
    Case{"Floor", MathOp::Floor, -100.0, 100.0, true},
    Case{"Ceil", MathOp::Ceil, -100.0, 100.0, true},
    Case{"Trunc", MathOp::Trunc, -100.0, 100.0, true},
    Case{"Round", MathOp::Round, -100.0, 100.0, true},
    Case{"Rint", MathOp::Rint, -100.0, 100.0, true},
    Case{"Fabs", MathOp::Fabs, -100.0, 100.0, true},
    Case{"Hypot", MathOp::Hypot, -100.0, 100.0, false},
    Case{"Copysign", MathOp::Copysign, -100.0, 100.0, true},
    Case{"Fmod", MathOp::Fmod, 0.5, 100.0, true},
    Case{"Fmin", MathOp::Fmin, -100.0, 100.0, true},
    Case{"Fmax", MathOp::Fmax, -100.0, 100.0, true},
    Case{"Fma", MathOp::Fma, -100.0, 100.0, true},
};

// The host reference, through std:: (see the file comment). rsqrt has no std::
// counterpart; 1/sqrt is the definition, compared within tolerance.
template <typename T>
T reference(const MathOp op, const T x, const T y, const T z) {
  switch (op) {
  case MathOp::Exp:
    return std::exp(x);
  case MathOp::Exp2:
    return std::exp2(x);
  case MathOp::Expm1:
    return std::expm1(x);
  case MathOp::Log:
    return std::log(x);
  case MathOp::Log2:
    return std::log2(x);
  case MathOp::Log10:
    return std::log10(x);
  case MathOp::Log1p:
    return std::log1p(x);
  case MathOp::Pow:
    return std::pow(x, y);
  case MathOp::Sqrt:
    return std::sqrt(x);
  case MathOp::Rsqrt:
    return T{1} / std::sqrt(x);
  case MathOp::Sin:
    return std::sin(x);
  case MathOp::Cos:
    return std::cos(x);
  case MathOp::Tan:
    return std::tan(x);
  case MathOp::Asin:
    return std::asin(x);
  case MathOp::Acos:
    return std::acos(x);
  case MathOp::Atan:
    return std::atan(x);
  case MathOp::Atan2:
    return std::atan2(x, y);
  case MathOp::Sinh:
    return std::sinh(x);
  case MathOp::Cosh:
    return std::cosh(x);
  case MathOp::Tanh:
    return std::tanh(x);
  case MathOp::Floor:
    return std::floor(x);
  case MathOp::Ceil:
    return std::ceil(x);
  case MathOp::Trunc:
    return std::trunc(x);
  case MathOp::Round:
    return std::round(x);
  case MathOp::Rint:
    return std::rint(x);
  case MathOp::Fabs:
    return std::fabs(x);
  case MathOp::Hypot:
    return std::hypot(x, y);
  case MathOp::Copysign:
    return std::copysign(x, y);
  case MathOp::Fmod:
    return std::fmod(x, y);
  case MathOp::Fmin:
    return std::fmin(x, y);
  case MathOp::Fmax:
    return std::fmax(x, y);
  case MathOp::Fma:
    return std::fma(x, y, z);
  case MathOp::IsNan:
    return std::isnan(x) ? T{1} : T{0};
  case MathOp::IsInf:
    return std::isinf(x) ? T{1} : T{0};
  case MathOp::IsFinite:
    return std::isfinite(x) ? T{1} : T{0};
  }
  return T{0};
}

// Seeded per case, so a failure reproduces and each case gets its own inputs.
template <typename T>
std::vector<T> draw(const Case &c, const std::uint32_t salt) {
  std::mt19937 rng{(static_cast<std::uint32_t>(c.op) * 7919U) + salt};
  std::uniform_real_distribution<double> dist{c.lo, c.hi};
  std::vector<T> v(kCount);
  for (T &e : v) {
    e = static_cast<T>(dist(rng));
  }
  return v;
}

template <typename T>
void check_case(const Case &c) {
  const std::vector<T> x = draw<T>(c, 1);
  const std::vector<T> y = draw<T>(c, 2);
  const std::vector<T> z = draw<T>(c, 3);
  std::vector<T> out(kCount);
  ASSERT_TRUE(tk::evaluate_on_device(c.op, x.data(), y.data(), z.data(), out.data(), kCount));

  const double eps = std::numeric_limits<T>::epsilon();
  std::size_t failures = 0;
  for (std::size_t i = 0; i < kCount && failures < 5; ++i) {
    const T expected = reference(c.op, x[i], y[i], z[i]);
    if (c.exact) {
      if (std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>(out[i]) !=
          std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>(
              expected)) {
        ++failures;
        ADD_FAILURE() << c.name << " not exact at x=" << x[i] << " y=" << y[i] << " z=" << z[i]
                      << ": device " << out[i] << ", host " << expected;
      }
      continue;
    }
    // Relative to the result, floored at 1 so results near zero (sin at a
    // multiple of pi, expm1 at 0) are judged in absolute terms instead.
    const double scale = std::max(std::fabs(static_cast<double>(expected)), 1.0);
    const double gap = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected));
    if (gap > kUlpTolerance * eps * scale) {
      ++failures;
      ADD_FAILURE() << c.name << " off by " << (gap / (eps * scale)) << " ulp at x=" << x[i]
                    << " y=" << y[i] << " z=" << z[i] << ": device " << out[i] << ", host "
                    << expected;
    }
  }
}

class PrimitivesGpu : public ::testing::TestWithParam<Case> {};

TEST_P(PrimitivesGpu, FloatMatchesHost) {
  FREE_BREAKFAST_REQUIRE_GPU();
  check_case<float>(GetParam());
}

TEST_P(PrimitivesGpu, DoubleMatchesHost) {
  FREE_BREAKFAST_REQUIRE_GPU();
  check_case<double>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Math, PrimitivesGpu, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<Case> &info) {
                           return std::string{info.param.name};
                         });

// The predicates, on the values that distinguish them -- random draws would
// never produce a NaN or an infinity.
template <typename T>
void check_predicates() {
  using L = std::numeric_limits<T>;
  const std::vector<T> x{T{0},           T{1},           T{-2.5},  L::infinity(),
                         -L::infinity(), L::quiet_NaN(), L::max(), L::denorm_min()};
  const std::vector<T> unused(x.size(), T{0});
  for (const MathOp op : {MathOp::IsNan, MathOp::IsInf, MathOp::IsFinite}) {
    std::vector<T> out(x.size());
    ASSERT_TRUE(
        tk::evaluate_on_device(op, x.data(), unused.data(), unused.data(), out.data(), x.size()));
    for (std::size_t i = 0; i < x.size(); ++i) {
      EXPECT_EQ(out[i], reference(op, x[i], T{0}, T{0}))
          << "op " << static_cast<std::int32_t>(op) << " at " << x[i];
    }
  }
}

TEST(PrimitivesGpuPredicates, Float) {
  FREE_BREAKFAST_REQUIRE_GPU();
  check_predicates<float>();
}

TEST(PrimitivesGpuPredicates, Double) {
  FREE_BREAKFAST_REQUIRE_GPU();
  check_predicates<double>();
}

// fmin/fmax with one NaN operand return the other -- the property that makes
// them more than `x < y ? x : y`, and one a device library could get wrong.
TEST(PrimitivesGpuPredicates, FminFmaxIgnoreOneNaN) {
  FREE_BREAKFAST_REQUIRE_GPU();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> x{nan, 1.0};
  const std::vector<double> y{1.0, nan};
  const std::vector<double> z(2, 0.0);
  for (const MathOp op : {MathOp::Fmin, MathOp::Fmax}) {
    std::vector<double> out(2);
    ASSERT_TRUE(tk::evaluate_on_device(op, x.data(), y.data(), z.data(), out.data(), 2));
    EXPECT_EQ(out[0], 1.0);
    EXPECT_EQ(out[1], 1.0);
  }
}

// The combinatorics are one constexpr definition on both sides, so the device
// result must equal the host's exactly -- and the host's is itself pinned by
// core_vocabulary_host.cpp's static_asserts.
TEST(PrimitivesGpuCombinatorics, MatchHostExactly) {
  FREE_BREAKFAST_REQUIRE_GPU();
  std::vector<std::int32_t> n;
  std::vector<std::int32_t> k;
  for (std::int32_t a = -2; a <= 20; ++a) {
    for (std::int32_t b = -1; b <= a + 1; ++b) {
      n.push_back(a);
      k.push_back(b);
    }
  }
  const auto check = [&](const tk::CombinatoricOp op, const auto host) {
    std::vector<double> out(n.size());
    ASSERT_TRUE(tk::combinatoric_on_device(op, n.data(), k.data(), out.data(), n.size()));
    for (std::size_t i = 0; i < n.size(); ++i) {
      EXPECT_EQ(out[i], host(n[i], k[i])) << "n=" << n[i] << " k=" << k[i];
    }
  };
  check(tk::CombinatoricOp::DoubleFactorialOdd,
        [](const std::int32_t a, std::int32_t) { return fb::math::double_factorial_odd(a); });
  check(tk::CombinatoricOp::Factorial,
        [](const std::int32_t a, std::int32_t) { return fb::math::factorial(a); });
  check(tk::CombinatoricOp::Binomial,
        [](const std::int32_t a, const std::int32_t b) { return fb::math::binomial(a, b); });
}

} // namespace
