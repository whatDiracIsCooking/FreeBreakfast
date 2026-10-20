// Host-only cases for the SAD initial guess: the aufbau order and per-subshell
// occupations, and one atom's spherical diagonal. The assembled guess over a
// placed basis is sad_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.sad;

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

fb::Shell shell_at_origin(std::int32_t l) {
  return fb::Shell(l, {{1.0, 1.0}}, kOrigin);
}

double sum(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

TEST(Sad, AufbauOrderSeatsOganesson) {
  std::int32_t total = 0;
  for (const fb::SubshellFill &subshell : fb::kAufbauOrder) {
    EXPECT_EQ(subshell.capacity, 2 * ((2 * subshell.l) + 1));
    total += subshell.capacity;
  }
  EXPECT_EQ(total, 118);
}

TEST(Sad, SubshellOccupationFillsInMadelungOrder) {
  EXPECT_TRUE(fb::subshell_occupation(0).empty());
  EXPECT_EQ(fb::subshell_occupation(1), (std::vector<std::int32_t>{1}));
  EXPECT_EQ(fb::subshell_occupation(6), (std::vector<std::int32_t>{2, 2, 2}));
  // Fe: 1s2 2s2 2p6 3s2 3p6 4s2 3d6 -- 4s before 3d.
  EXPECT_EQ(fb::subshell_occupation(26), (std::vector<std::int32_t>{2, 2, 6, 2, 6, 2, 6}));
  const std::vector<std::int32_t> og = fb::subshell_occupation(118);
  EXPECT_EQ(og.size(), fb::kAufbauOrder.size());
  EXPECT_EQ(std::accumulate(og.begin(), og.end(), 0), 118);
}

TEST(SadDeathTest, SubshellOccupationAbortsOutsideTheTable) {
  EXPECT_DEATH((void)fb::subshell_occupation(-1), "");
  EXPECT_DEATH((void)fb::subshell_occupation(119), "");
}

TEST(Sad, AtomicDiagonalSpreadsEachSubshellOverItsShell) {
  // O over s, s, p: 1s2 on the first s, 2s2 on the second, 2p4 over three m.
  const std::vector<fb::Shell> oxygen{shell_at_origin(0), shell_at_origin(0), shell_at_origin(1)};
  const std::vector<double> diag = fb::atomic_diagonal(8, oxygen);
  ASSERT_EQ(diag.size(), 5U);
  EXPECT_EQ(diag[0], 2.0);
  EXPECT_EQ(diag[1], 2.0);
  for (std::size_t m = 2; m < 5; ++m) {
    EXPECT_DOUBLE_EQ(diag[m], 4.0 / 3.0);
  }
  EXPECT_DOUBLE_EQ(sum(diag), 8.0);
}

TEST(Sad, AtomicDiagonalIsSphericalAndMatchesShellsByMomentum) {
  // Fe with its shells deliberately out of aufbau order: the d shell first.
  // The k-th subshell of momentum l still lands on the k-th shell of that l,
  // and 3d6 spreads over 5 spherical functions, not 6 Cartesian ones.
  const std::vector<fb::Shell> iron{shell_at_origin(2), shell_at_origin(0), shell_at_origin(0),
                                    shell_at_origin(1), shell_at_origin(0), shell_at_origin(1),
                                    shell_at_origin(0)};
  const std::vector<double> diag = fb::atomic_diagonal(26, iron);
  ASSERT_EQ(diag.size(), 15U);
  for (std::size_t m = 0; m < 5; ++m) {
    EXPECT_DOUBLE_EQ(diag[m], 6.0 / 5.0);
  }
  EXPECT_EQ(diag[5], 2.0);  // 1s
  EXPECT_EQ(diag[6], 2.0);  // 2s
  EXPECT_EQ(diag[7], 2.0);  // 2p, 6 over 3
  EXPECT_EQ(diag[10], 2.0); // 3s
  EXPECT_EQ(diag[11], 2.0); // 3p
  EXPECT_EQ(diag[14], 2.0); // 4s
  EXPECT_DOUBLE_EQ(sum(diag), 26.0);
}

TEST(Sad, AtomicDiagonalDropsSubshellsTheBasisCannotSeat) {
  // C over a single s shell: 1s2 seats, 2s2 and 2p2 have nowhere to go.
  const std::vector<fb::Shell> carbon{shell_at_origin(0)};
  EXPECT_EQ(fb::atomic_diagonal(6, carbon), (std::vector<double>{2.0}));
  // An unoccupied polarization shell stays zero.
  const std::vector<fb::Shell> hydrogen{shell_at_origin(0), shell_at_origin(1)};
  EXPECT_EQ(fb::atomic_diagonal(1, hydrogen), (std::vector<double>{1.0, 0.0, 0.0, 0.0}));
}

} // namespace
