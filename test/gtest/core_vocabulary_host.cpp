// Host-only cases for core/types (including its :concepts partition),
// core/constants and core/nontransferrable. Most of what these modules promise is compile-time,
// so most of the checks are static_asserts; the TESTs exist so each module has
// a ctest entry and the vector types are exercised as values, not just names.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import wwr.vector_types;

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
static_assert(std::is_same_v<fb::Vec<int, 2>, wwr::int2>);
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
static_assert(!fb::real_fp<int>);
static_assert(!fb::real_fp<long double>);

// int_type is wwr's: exactly int and int64_t. size_t and double are both out.
static_assert(fb::int_type<int>);
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
static_assert(fb::kTwo<int> == 2);
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
