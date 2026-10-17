// Host-only cases for core/types (including its :concepts partition),
// core/constants, core/primitives, core/kernel and core/nontransferrable. Most
// of what these modules promise is compile-time, so most of the checks are
// static_asserts; the TESTs exist so each module has a ctest entry and the
// vector types are exercised as values, not just names.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------

// The :integral_types partition exports the fixed-width types from free_breakfast.
static_assert(std::is_same_v<fb::int32_t, std::int32_t>);
static_assert(std::is_same_v<fb::uint64_t, std::uint64_t>);
static_assert(std::is_same_v<fb::size_t, std::size_t>);

// Vec maps onto the builtins on either backend. The builtin names themselves
// arrive through types.cppm's global module fragment, so an importer cannot
// spell ::float3 -- compare against wwr's exported aliases, which are the same
// types.
static_assert(std::is_same_v<fb::Vec<float, 3>, wwr::float3>);
static_assert(std::is_same_v<fb::Vec<std::int32_t, 2>, wwr::int2>);
// int-type-ok: long long is distinct from int64_t (long on LP64), and Vec maps both.
static_assert(std::is_same_v<fb::Vec<long long, 1>, wwr::longlong1>);
static_assert(std::is_same_v<fb::Vec<unsigned char, 4>, wwr::uchar4>);

// wwr has no double4 (CUDA 13 deprecates it), so check the one Vec keeps by
// shape: four doubles. Spelled out each time rather than aliased: a local
// `using D4 = fb::Vec<double, 4>;` re-attaches the deprecation warning that
// naming Vec directly avoids.
static_assert(sizeof(fb::Vec<double, 4>) == 4 * sizeof(double));
static_assert(std::is_same_v<decltype(fb::Vec<double, 4>{}.w), double>);

// make_vec's arity picks the order.
static_assert(std::is_same_v<decltype(fb::make_vec<double>(1, 2, 3, 4)), fb::Vec<double, 4>>);
static_assert(std::is_same_v<decltype(fb::make_vec<float>(1, 2)), wwr::float2>);

TEST(CoreTypes, MakeVecConvertsEachComponent) {
  const auto v = fb::make_vec<double>(1, 2.5f, 3u, -4);
  EXPECT_EQ(v.x, 1.0);
  EXPECT_EQ(v.y, 2.5);
  EXPECT_EQ(v.z, 3.0);
  EXPECT_EQ(v.w, -4.0);

  const auto f = fb::make_vec<float>(0.5, 7);
  EXPECT_EQ(f.x, 0.5f);
  EXPECT_EQ(f.y, 7.0f);
}

// ---------------------------------------------------------------------------
// types:concepts
// ---------------------------------------------------------------------------

static_assert(fb::real_fp<float>);
static_assert(fb::real_fp<double>);
static_assert(!fb::real_fp<std::int32_t>);
static_assert(!fb::real_fp<long double>);

// int_type is wwr's: exactly int and int64_t. size_t and double are both out.
static_assert(fb::int_type<std::int32_t>);
static_assert(fb::int_type<std::int64_t>);
static_assert(!fb::int_type<std::size_t>);
static_assert(!fb::int_type<double>);
static_assert(!fb::int_type<bool>);

template <fb::real_fp T>
constexpr T twice(T x) {
  return x + x;
}

TEST(CoreConcepts, ConstrainTemplates) {
  EXPECT_EQ(twice(1.5), 3.0);
  EXPECT_EQ(twice(1.5f), 3.0f);
}

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

static_assert(fb::kZero<double> == 0.0);
static_assert(fb::kOne<float> == 1.0f);
static_assert(fb::kTwo<std::int32_t> == 2);
static_assert(fb::kNegOne<double> == -1.0);
static_assert(fb::kKilo<double> == 1e3);
static_assert(fb::kFemto<double> == 1e-15);

TEST(CoreConstants, BohrConversionsAreReciprocal) {
  EXPECT_DOUBLE_EQ(fb::kAngstromPerBohr<double> * fb::kBohrPerAngstrom<double>, 1.0);
  EXPECT_DOUBLE_EQ(fb::kAngstromPerBohr<double>, 0.529177210903);
}

TEST(CoreConstants, FaradayIsAvogadroTimesElementaryCharge) {
  // Both factors are exact in the 2019 SI, so F = N_A * e to double precision.
  EXPECT_NEAR(fb::kAvogadro<double> * fb::kElementaryCharge<double>, fb::kFaraday<double>,
              1e-9 * fb::kFaraday<double>);
  EXPECT_NEAR(fb::kBoltzmann<double> * fb::kAvogadro<double>, fb::kGasConstant<double>,
              1e-12 * fb::kGasConstant<double>);
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

// Each function keeps the argument's precision rather than promoting to double.
static_assert(std::is_same_v<decltype(fb::math::sqrt(2.0f)), float>);
static_assert(std::is_same_v<decltype(fb::math::sqrt(2.0)), double>);
static_assert(std::is_same_v<decltype(fb::math::fma(1.0f, 2.0f, 3.0f)), float>);
static_assert(std::is_same_v<decltype(fb::math::isnan(1.0)), bool>);

// real_fp admits exactly float and double, matching the device templates'
// static_assert: an int is rejected rather than promoted, as is long double.
// (A concept, not a bare `requires { ... }`: outside a template the expression is
// not dependent, so an ill-formed call is a hard error rather than false.)
template <typename T>
concept HasMathSqrt = requires(T x) { fb::math::sqrt(x); };
template <typename T, typename U>
concept HasMathPow = requires(T x, U y) { fb::math::pow(x, y); };

static_assert(HasMathSqrt<float> && HasMathSqrt<double>);
static_assert(!HasMathSqrt<std::int32_t>);
static_assert(!HasMathSqrt<long double>);
static_assert(!HasMathPow<double, float>);

// The combinatorics are constexpr: the same definition a device TU compiles.
static_assert(fb::math::double_factorial_odd(-1) == 1.0);
static_assert(fb::math::double_factorial_odd(1) == 1.0);
static_assert(fb::math::double_factorial_odd(7) == 105.0);
static_assert(fb::math::factorial(0) == 1.0);
static_assert(fb::math::factorial(5) == 120.0);
static_assert(fb::math::binomial(5, 2) == 10.0);
static_assert(fb::math::binomial(4, 0) == 1.0);
static_assert(fb::math::binomial(4, 5) == 0.0);
static_assert(fb::math::binomial(4, -1) == 0.0);

TEST(CorePrimitives, CombinatoricsAgree) {
  // (2n-1)!! = (2n)! / (2^n n!), and the binomial row sums to 2^n.
  for (std::int32_t n = 0; n <= 10; ++n) {
    EXPECT_EQ(fb::math::double_factorial_odd((2 * n) - 1),
              fb::math::factorial(2 * n) / (std::ldexp(1.0, n) * fb::math::factorial(n)))
        << "n = " << n;
    double row = 0.0;
    for (std::int32_t k = 0; k <= n; ++k) {
      row += fb::math::binomial(n, k);
    }
    EXPECT_EQ(row, std::ldexp(1.0, n)) << "n = " << n;
  }
}

// Integer rounding keeps the argument's type, so a size_t byte count is not
// narrowed, and is constexpr like the combinatorics.
static_assert(
    std::is_same_v<decltype(fb::math::idivup(std::size_t{7}, std::size_t{2})), std::size_t>);
static_assert(fb::math::idivup(0, 4) == 0);
static_assert(fb::math::idivup(1, 4) == 1);
static_assert(fb::math::idivup(8, 4) == 2);
static_assert(fb::math::idivup(9, 4) == 3);
static_assert(fb::math::align_up(0, 4) == 0);
static_assert(fb::math::align_up(9, 4) == 12);
static_assert(fb::math::align_up(12, 4) == 12);

template <typename T>
concept HasIdivup = requires(T x) { fb::math::idivup(x, x); };

static_assert(HasIdivup<std::int32_t> && HasIdivup<std::uint64_t>);
static_assert(!HasIdivup<double>);

TEST(CorePrimitives, IntegerRoundingCoversEveryRemainder) {
  for (std::uint32_t a = 0; a <= 100; ++a) {
    for (std::uint32_t b = 1; b <= 9; ++b) {
      const std::uint32_t q = fb::math::idivup(a, b);
      const std::uint32_t r = fb::math::align_up(a, b);
      // q is the smallest count of b-chunks covering a; r is that many b's.
      EXPECT_GE(q * b, a) << a << " / " << b;
      EXPECT_LT((q * b) - a, b) << a << " / " << b;
      EXPECT_EQ(r, q * b) << a << " / " << b;
    }
  }
}

TEST(CorePrimitives, MatchCmath) {
  EXPECT_EQ(fb::math::sqrt(2.0), std::sqrt(2.0));
  EXPECT_EQ(fb::math::sqrt(2.0f), std::sqrt(2.0f));
  EXPECT_EQ(fb::math::log(10.0), std::log(10.0));
  EXPECT_EQ(fb::math::exp(-0.5), std::exp(-0.5));
  EXPECT_EQ(fb::math::sin(0.25), std::sin(0.25));
  EXPECT_EQ(fb::math::cos(0.25f), std::cos(0.25f));
  EXPECT_EQ(fb::math::atan2(1.0, -1.0), std::atan2(1.0, -1.0));
  EXPECT_EQ(fb::math::pow(2.0, 0.5), std::pow(2.0, 0.5));
  EXPECT_EQ(fb::math::fma(2.0, 3.0, 4.0), 10.0);
  EXPECT_EQ(fb::math::round(2.5), 3.0);
  EXPECT_DOUBLE_EQ(fb::math::rsqrt(4.0), 0.5);
}

TEST(CorePrimitives, NaNHandling) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(fb::math::isnan(nan));
  EXPECT_FALSE(fb::math::isfinite(nan));
  EXPECT_TRUE(fb::math::isinf(std::numeric_limits<double>::infinity()));
  // fmin/fmax return the non-NaN operand, which `x < y ? x : y` would not.
  EXPECT_EQ(fb::math::fmin(nan, 1.0), 1.0);
  EXPECT_EQ(fb::math::fmax(1.0, nan), 1.0);
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

static_assert(std::is_same_v<std::remove_const_t<decltype(fb::kWarpSize)>, std::uint32_t>);
static_assert(fb::kWarpSize == 32 || fb::kWarpSize == 64);
static_assert(std::is_same_v<std::remove_const_t<decltype(fb::kFourWarps)>, std::uint32_t>);
static_assert(fb::kFourWarps == 4 * fb::kWarpSize && fb::NumWarps<4>);

// NumWarps constrains a value: >= 1, a power of two, and a block of at most
// 1024 threads -- checked without overflowing for a 64-bit count.
static_assert(fb::NumWarps<1> && fb::NumWarps<2U> && fb::NumWarps<1024 / fb::kWarpSize>);
static_assert(!fb::NumWarps<0> && !fb::NumWarps<-1> && !fb::NumWarps<3>);
static_assert(!fb::NumWarps<2048 / fb::kWarpSize>);
static_assert(!fb::NumWarps<std::uint64_t{1} << 32>);
static_assert(!fb::NumWarps<1.0>);

TEST(CoreKernel, WarpSizeDividesTheBlockCap) {
  EXPECT_EQ(1024 % fb::kWarpSize, 0U);
}

// ---------------------------------------------------------------------------
// nontransferrable
// ---------------------------------------------------------------------------

struct OnlyMoves : private fb::NonCopyable {};
struct OnlyCopies : private fb::NonMovable {};
struct Pinned : private fb::NonTransferable {};

static_assert(!std::is_copy_constructible_v<OnlyMoves>);
static_assert(std::is_move_constructible_v<OnlyMoves>);
static_assert(std::is_copy_constructible_v<OnlyCopies>);
static_assert(!std::copyable<Pinned>);
static_assert(!std::movable<Pinned>);

TEST(CoreNontransferrable, DerivedTypesStillDefaultConstruct) {
  [[maybe_unused]] OnlyMoves a;
  [[maybe_unused]] OnlyMoves b = std::move(a);
  [[maybe_unused]] OnlyCopies c;
  [[maybe_unused]] OnlyCopies d = c;
  [[maybe_unused]] Pinned p;
  SUCCEED();
}

} // namespace
