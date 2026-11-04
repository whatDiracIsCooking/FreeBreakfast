// Host-only cases for the Optimize step's state following:
// normalized_overlaps, the root overlaps each over its row's norm; match_roots,
// which matches the last geometry's CASSCF roots to this one's by their
// overlaps, follows a state-averaged root through a root swap and refuses,
// Numerical, a followed root whose normalized overlap drops below
// kStateOverlapFloor; and qd_state_overlap, which carries u_K across in that
// matched basis, follows the QD state through a root swap and a sign flip,
// and refuses when its overlap drops below the floor (the state lost at a
// crossing) or the roots do not match one to one; and pair_overlap, which
// keeps an MECI pair through a swap of I and J (and any turn within the
// pair) and refuses, Numerical, a pair whose 2 x 2 block's smaller singular
// value falls below the floor; and its QD form over H_eff's eigenvectors,
// carried through the same overlaps, which keeps a swapped P/Q block and
// refuses one whose smaller singular value is under the floor. Nothing is
// run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// The identity: the roots carried into themselves.
constexpr std::array<double, 4> kSameRoots = {1.0, 0.0, 0.0, 1.0};

// u turned by @p degrees from @p u, two roots.
std::array<double, 2> turned(std::array<double, 2> u, double degrees) {
  const double a = degrees * std::numbers::pi / 180.0;
  return {(std::cos(a) * u[0]) - (std::sin(a) * u[1]), (std::sin(a) * u[0]) + (std::cos(a) * u[1])};
}

// A small turn is the same state, at cos of the angle, and u's own sign is
// arbitrary.
TEST(PipelineOptimize, AQdStateTurnedALittleIsFollowed) {
  const std::array<double, 2> u = {0.8, 0.6};
  const auto small = fb::qd_state_overlap(kSameRoots, u, turned(u, 5.0));
  ASSERT_TRUE(small.has_value()) << small.error().message;
  EXPECT_NEAR(*small, std::cos(5.0 * std::numbers::pi / 180.0), 1e-15);
  const std::array<double, 2> flipped = {-u[0], -u[1]};
  const auto sign = fb::qd_state_overlap(kSameRoots, u, flipped);
  ASSERT_TRUE(sign.has_value()) << sign.error().message;
  EXPECT_NEAR(*sign, 1.0, 1e-15);
}

// The CASSCF roots swap order between the geometries, one with its CI phase
// flipped: <0(last)|1(this)> = -1, <1(last)|0(this)> = 1. The same state's u
// over this geometry's roots is the last u permuted and signed, and it is
// matched in full, where an overlap taken by index would be 0.
TEST(PipelineOptimize, AQdStateIsFollowedThroughARootSwap) {
  const std::array<double, 4> swapped = {0.0, -1.0, 1.0, 0.0};
  const std::array<double, 2> last = {0.8, 0.6};
  const std::array<double, 2> now = {0.6, -0.8};
  const auto overlap = fb::qd_state_overlap(swapped, last, now);
  ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
  EXPECT_NEAR(*overlap, 1.0, 1e-15);
  // By index the two are orthogonal: this is the other state.
  const auto by_index = fb::qd_state_overlap(kSameRoots, last, now);
  ASSERT_FALSE(by_index.has_value());
  EXPECT_EQ(by_index.error().kind, fb::ErrorKind::Numerical);
}

// The refusal: an overlap at the floor is followed, one just below it is
// Numerical, naming the overlap, and so is a turn of 30 degrees (cos 30 =
// 0.866), the state lost to another at a crossing rather than followed onto
// its surface.
TEST(PipelineOptimize, AQdStateBelowTheOverlapFloorIsRefusedAsNumerical) {
  const std::array<double, 2> last = {1.0, 0.0};
  const double floor = fb::kStateOverlapFloor;
  const std::array<double, 2> at = {floor, std::sqrt(1.0 - (floor * floor))};
  const auto kept = fb::qd_state_overlap(kSameRoots, last, at);
  ASSERT_TRUE(kept.has_value()) << kept.error().message;
  EXPECT_EQ(*kept, floor);

  const double below = std::nextafter(floor, 0.0);
  const std::array<double, 2> under = {below, std::sqrt(1.0 - (below * below))};
  const auto lost = fb::qd_state_overlap(kSameRoots, last, under);
  ASSERT_FALSE(lost.has_value());
  EXPECT_EQ(lost.error().kind, fb::ErrorKind::Numerical) << lost.error().message;
  EXPECT_NE(lost.error().message.find("qd_state_overlap: the QD state's overlap with the last "
                                      "geometry's is 0.900000, below 0.9"),
            std::string::npos)
      << lost.error().message;

  const auto crossing = fb::qd_state_overlap(kSameRoots, last, turned(last, 30.0));
  ASSERT_FALSE(crossing.has_value());
  EXPECT_EQ(crossing.error().kind, fb::ErrorKind::Numerical) << crossing.error().message;
  EXPECT_NE(crossing.error().message.find("is 0.866025, below 0.9: it is lost"), std::string::npos)
      << crossing.error().message;
}

// Two last roots both nearest the same root of this geometry: the roots are
// not followed, Numerical before any u is compared.
TEST(PipelineOptimize, QdRootsThatDoNotMatchOneToOneAreRefused) {
  const std::array<double, 4> both_to_one = {0.6, 0.8, 0.5, 0.86};
  const std::array<double, 2> u = {1.0, 0.0};
  const auto refused = fb::qd_state_overlap(both_to_one, u, u);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
  EXPECT_NE(refused.error().message.find("root 1 matches no root of this one alone"),
            std::string::npos)
      << refused.error().message;
}

TEST(PipelineOptimize, QdStateOverlapSizesThatDoNotAgreeAreInvalidConfig) {
  const std::array<double, 2> u = {1.0, 0.0};
  const std::array<double, 3> three = {1.0, 0.0, 0.0};
  for (const auto &[s, a, b] :
       {std::tuple{std::span<const double>(kSameRoots), std::span<const double>(u),
                   std::span<const double>(three)},
        std::tuple{std::span<const double>(u), std::span<const double>(u),
                   std::span<const double>(u)},
        std::tuple{std::span<const double>{}, std::span<const double>{},
                   std::span<const double>{}}}) {
    const auto refused = fb::qd_state_overlap(s, a, b);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
  }
}

// A state-averaged root followed through a swap of the CASSCF roots, under
// the overlaps' common decay with the step (every element scaled by 0.7):
// the last root 1 is this geometry's root 0, its overlap normalized by its
// row, and root 0 is root 1 with its phase flipped.
TEST(PipelineOptimize, ARootIsFollowedThroughARootSwap) {
  constexpr double kDecay = 0.7;
  const std::array<double, 4> swapped = {kDecay * 0.06, kDecay * -0.98, kDecay * 0.97,
                                         kDecay * 0.05};
  const std::array<std::int32_t, 1> followed = {1};
  const auto match = fb::match_roots(swapped, followed);
  ASSERT_TRUE(match.has_value()) << match.error().message;
  ASSERT_EQ(match->size(), 2U);
  EXPECT_EQ((*match)[1].root, 0);
  EXPECT_NEAR((*match)[1].overlap, 0.97 / std::hypot(0.97, 0.05), 1e-15);
  EXPECT_EQ((*match)[0].root, 1);
  EXPECT_NEAR((*match)[0].overlap, -0.98 / std::hypot(0.06, 0.98), 1e-15);
}

// The normalized overlaps match_roots matches over, and a pair check reads a
// block of: each row over its 2-norm, the common decay gone, a row of zeros
// refused.
TEST(PipelineOptimize, NormalizedOverlapsDivideEachRowByItsNorm) {
  constexpr double kDecay = 0.5;
  const std::array<double, 4> decayed = {kDecay * 0.6, kDecay * 0.8, kDecay * -0.8, kDecay * 0.6};
  const auto s = fb::normalized_overlaps(decayed);
  ASSERT_TRUE(s.has_value()) << s.error().message;
  const std::array<double, 4> want = {0.6, 0.8, -0.8, 0.6};
  for (std::size_t k = 0; k < want.size(); ++k) {
    EXPECT_NEAR((*s)[k], want[k], 1e-15) << k;
  }
  const std::array<double, 4> zero_row = {0.0, 0.0, 0.0, 1.0};
  const auto refused = fb::normalized_overlaps(zero_row);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
}

// The refusal: a followed root turned by 20 degrees (cos 20 = 0.940) is
// kept; one turned by 30 (cos 30 = 0.866) is Numerical, naming the root, its
// match and the overlap, though the roots still match one to one and are
// matched when nothing is followed.
TEST(PipelineOptimize, ARootBelowTheOverlapFloorIsRefusedAsNumerical) {
  const std::array<std::int32_t, 1> root_zero = {0};
  const double a = 20.0 * std::numbers::pi / 180.0;
  const std::array<double, 4> twenty = {std::cos(a), std::sin(a), -std::sin(a), std::cos(a)};
  const auto kept = fb::match_roots(twenty, root_zero);
  ASSERT_TRUE(kept.has_value()) << kept.error().message;
  EXPECT_EQ((*kept)[0].root, 0);
  EXPECT_NEAR((*kept)[0].overlap, std::cos(a), 1e-15);
  EXPECT_GE((*kept)[0].overlap, fb::kStateOverlapFloor);

  // A 30 degree turn: cos 30 = 0.866.
  const double c = std::cos(std::numbers::pi / 6.0);
  const double s = std::sin(std::numbers::pi / 6.0);
  const std::array<double, 4> turned_roots = {c, s, -s, c};
  ASSERT_TRUE(fb::match_roots(turned_roots).has_value());
  const auto lost = fb::match_roots(turned_roots, root_zero);
  ASSERT_FALSE(lost.has_value());
  EXPECT_EQ(lost.error().kind, fb::ErrorKind::Numerical) << lost.error().message;
  EXPECT_NE(lost.error().message.find("match_roots: the last geometry's CASSCF root 0 has "
                                      "overlap 0.866025 with its match, root 0, below 0.9"),
            std::string::npos)
      << lost.error().message;
}

// Sizes that are no n x n, and a followed root that is not one of the n,
// are InvalidConfig.
TEST(PipelineOptimize, MatchRootsSizesAndRootsOutOfRangeAreInvalidConfig) {
  const std::array<double, 3> three = {1.0, 0.0, 0.0};
  const auto not_square = fb::match_roots(three);
  ASSERT_FALSE(not_square.has_value());
  EXPECT_EQ(not_square.error().kind, fb::ErrorKind::InvalidConfig);
  const auto empty = fb::match_roots(std::span<const double>{});
  ASSERT_FALSE(empty.has_value());
  EXPECT_EQ(empty.error().kind, fb::ErrorKind::InvalidConfig);
  const std::array<std::int32_t, 1> past = {2};
  const auto refused = fb::match_roots(kSameRoots, past);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
}

// --- An MECI pair -------------------------------------------------------------

// The pair's roots swap order, each with either phase, under the overlaps'
// common decay: the 2 x 2 block, row-normalized, is [[0, +-1], [+-1, 0]],
// both singular values 1, and the pair is kept. So is the pair turned
// within itself by 45 degrees, where match_roots would find no root to
// follow: I and J may mix freely, as the MECI re-takes the lower and upper
// by energy at every geometry.
TEST(PipelineOptimize, AnMeciPairIsKeptThroughASwapOfItsRoots) {
  constexpr double kDecay = 0.6;
  for (const double first : {1.0, -1.0}) {
    for (const double second : {1.0, -1.0}) {
      const std::array<double, 4> swapped = {0.0, kDecay * first, kDecay * second, 0.0};
      const auto overlap = fb::pair_overlap(swapped, 0, 1);
      ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
      EXPECT_NEAR(*overlap, 1.0, 1e-15) << first << " " << second;
      // The pair named in either order.
      const auto reversed = fb::pair_overlap(swapped, 1, 0);
      ASSERT_TRUE(reversed.has_value()) << reversed.error().message;
      EXPECT_NEAR(*reversed, 1.0, 1e-15);
    }
  }
  // Three roots, the pair (0, 2) swapped and root 1 kept.
  const std::array<double, 9> three = {0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
  const auto pair = fb::pair_overlap(three, 0, 2);
  ASSERT_TRUE(pair.has_value()) << pair.error().message;
  EXPECT_NEAR(*pair, 1.0, 1e-15);
  const double c = std::cos(std::numbers::pi / 4.0);
  const std::array<double, 4> mixed = {c, c, -c, c};
  ASSERT_FALSE(fb::match_roots(mixed).has_value());
  const auto within = fb::pair_overlap(mixed, 0, 1);
  ASSERT_TRUE(within.has_value()) << within.error().message;
  EXPECT_NEAR(*within, 1.0, 1e-15);
}

// Three roots, root 1 of the pair (0, 1) turning into root 2, which is
// outside it: the block is [[1, 0], [0, cos a]], its smaller singular value
// cos a. At 20 degrees (0.940) the pair is kept, at that value; at 30
// (0.866, below the floor) it is Numerical, naming the roots and the value.
TEST(PipelineOptimize, AnMeciPairBelowTheOverlapFloorIsRefusedAsNumerical) {
  const auto turned_out = [](double degrees) {
    const double a = degrees * std::numbers::pi / 180.0;
    return std::array<double, 9>{1.0,         0.0, 0.0,          0.0,        std::cos(a),
                                 std::sin(a), 0.0, -std::sin(a), std::cos(a)};
  };
  const auto kept = fb::pair_overlap(turned_out(20.0), 0, 1);
  ASSERT_TRUE(kept.has_value()) << kept.error().message;
  EXPECT_NEAR(*kept, std::cos(20.0 * std::numbers::pi / 180.0), 1e-15);
  EXPECT_GE(*kept, fb::kStateOverlapFloor);

  const auto lost = fb::pair_overlap(turned_out(30.0), 0, 1);
  ASSERT_FALSE(lost.has_value());
  EXPECT_EQ(lost.error().kind, fb::ErrorKind::Numerical) << lost.error().message;
  EXPECT_NE(lost.error().message.find("pair_overlap: the last geometry's CASSCF roots 0 and 1 "
                                      "keep 0.866025 of their space in this one's, below 0.9"),
            std::string::npos)
      << lost.error().message;
  // The pair (0, 2) is not the one turned out of: root 1 is outside it, and
  // root 2's row keeps cos 30 inside the block, so it is refused the same.
  const auto other = fb::pair_overlap(turned_out(30.0), 0, 2);
  ASSERT_FALSE(other.has_value());
  EXPECT_EQ(other.error().kind, fb::ErrorKind::Numerical);
}

// Sizes that are no n x n, and roots that are not two distinct ones of the
// n, are InvalidConfig.
TEST(PipelineOptimize, PairOverlapSizesAndRootsOutOfRangeAreInvalidConfig) {
  const std::array<double, 3> three = {1.0, 0.0, 0.0};
  for (const auto &[overlaps, i, j] : {std::tuple{std::span<const double>(three), 0, 1},
                                       std::tuple{std::span<const double>(kSameRoots), 0, 0},
                                       std::tuple{std::span<const double>(kSameRoots), 0, 2},
                                       std::tuple{std::span<const double>(kSameRoots), -1, 1}}) {
    const auto refused = fb::pair_overlap(overlaps, i, j);
    ASSERT_FALSE(refused.has_value()) << i << " " << j;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
  }
}

// --- A QD-NEVPT2 MECI pair ----------------------------------------------------

// R, n x n, state K's eigenvector at [K n, (K + 1) n): the identity, three
// roots.
constexpr std::array<double, 9> kIdentity3 = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

// The QD states P and Q swap, each with either phase, and the CASSCF roots
// under them stay put: the 2 x 2 block u_a(last)^T S u_b(this) is
// [[0, +-1], [+-1, 0]], both singular values 1, and the pair is kept, named in
// either order. So is the same swap seen through roots that swapped too,
// with the overlaps' common decay, and a pair turned 45 degrees within
// itself.
TEST(PipelineOptimize, AQdMeciPairIsKeptThroughASwapOfItsStates) {
  for (const double first : {1.0, -1.0}) {
    for (const double second : {1.0, -1.0}) {
      // This geometry's u_0 = second e_1, u_1 = first e_0, u_2 = e_2.
      const std::array<double, 9> swapped = {0.0, second, 0.0, first, 0.0, 0.0, 0.0, 0.0, 1.0};
      for (const auto &[p, q] : {std::pair{0, 1}, std::pair{1, 0}}) {
        const auto overlap = fb::pair_overlap(kIdentity3, kIdentity3, swapped, p, q);
        ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
        EXPECT_NEAR(*overlap, 1.0, 1e-15) << first << " " << second << " " << p;
      }
    }
  }
  // The roots 0 and 1 swapped (decayed by 0.6), and R swapped back over them:
  // the states have not moved.
  constexpr double kDecay = 0.6;
  const std::array<double, 9> roots = {0.0, kDecay, 0.0, -kDecay, 0.0, 0.0, 0.0, 0.0, kDecay};
  const std::array<double, 9> back = {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const auto through = fb::pair_overlap(roots, kIdentity3, back, 0, 1);
  ASSERT_TRUE(through.has_value()) << through.error().message;
  EXPECT_NEAR(*through, 1.0, 1e-15);
  const double c = std::cos(std::numbers::pi / 4.0);
  const std::array<double, 9> mixed = {c, c, 0.0, -c, c, 0.0, 0.0, 0.0, 1.0};
  const auto within = fb::pair_overlap(kIdentity3, kIdentity3, mixed, 0, 1);
  ASSERT_TRUE(within.has_value()) << within.error().message;
  EXPECT_NEAR(*within, 1.0, 1e-15);
}

// State 1 of the pair (0, 1) turning into state 2, outside it: the block is
// [[1, 0], [0, cos a]]. At 20 degrees (0.940) the pair is kept at that value;
// at 30 (0.866, under the floor) it is Numerical, naming the states.
TEST(PipelineOptimize, AQdMeciPairBelowTheOverlapFloorIsRefusedAsNumerical) {
  const auto turned_out = [](double degrees) {
    const double a = degrees * std::numbers::pi / 180.0;
    return std::array<double, 9>{1.0,         0.0, 0.0,          0.0,        std::cos(a),
                                 std::sin(a), 0.0, -std::sin(a), std::cos(a)};
  };
  const auto kept = fb::pair_overlap(kIdentity3, kIdentity3, turned_out(20.0), 0, 1);
  ASSERT_TRUE(kept.has_value()) << kept.error().message;
  EXPECT_NEAR(*kept, std::cos(20.0 * std::numbers::pi / 180.0), 1e-15);

  const auto lost = fb::pair_overlap(kIdentity3, kIdentity3, turned_out(30.0), 0, 1);
  ASSERT_FALSE(lost.has_value());
  EXPECT_EQ(lost.error().kind, fb::ErrorKind::Numerical) << lost.error().message;
  EXPECT_NE(lost.error().message.find("pair_overlap: the last geometry's QD states 0 and 1 keep "
                                      "0.866025 of their space in this one's, below 0.9"),
            std::string::npos)
      << lost.error().message;
}

// Eigenvectors of the wrong size, and states that are not two distinct ones
// of the n, are InvalidConfig.
TEST(PipelineOptimize, QdPairOverlapSizesAndStatesOutOfRangeAreInvalidConfig) {
  const std::array<double, 4> two = {1.0, 0.0, 0.0, 1.0};
  const std::span<const double> three(kIdentity3);
  for (const auto &[previous, current, p, q] :
       {std::tuple{std::span<const double>(two), three, 0, 1},
        std::tuple{three, std::span<const double>(two), 0, 1}, std::tuple{three, three, 1, 1},
        std::tuple{three, three, 0, 3}, std::tuple{three, three, -1, 0}}) {
    const auto refused = fb::pair_overlap(kIdentity3, previous, current, p, q);
    ASSERT_FALSE(refused.has_value()) << p << " " << q;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
  }
}

} // namespace
