// Host-only cases for the Optimize step's QD state following:
// qd_state_overlap, which matches the last geometry's CASSCF roots to this
// one's by their overlaps and carries u_K across in that matched basis. It
// follows the state through a root swap and a sign flip, and refuses,
// Numerical, when the matched overlap drops below kQdStateOverlapFloor (the
// state lost at a crossing) or the roots do not match one to one. Nothing
// is run, so nothing touches a device.

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
  const double floor = fb::kQdStateOverlapFloor;
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

} // namespace
