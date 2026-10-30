// Host-only cases for basis's :basis_set: the two AO layouts, the per-atom
// shell grouping, and the Cartesian-to-spherical transform matrix.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

#include "gaussian_moments.h"

namespace {

namespace fb = free_breakfast;
using fb::test::monomial_gram;
using fb::test::Powers;
using fb::test::row_of;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

TEST(BasisSet, EmptyBasisHasNoFunctions) {
  const fb::BasisSet basis;
  EXPECT_EQ(basis.nshells(), 0U);
  EXPECT_EQ(basis.nao_cart(), 0U);
  EXPECT_EQ(basis.nao_sphe(), 0U);
  EXPECT_TRUE(basis.aos_cart().empty());
  EXPECT_TRUE(fb::spherical_transform_matrix(basis).empty());
}

// Moving takes the offset vectors, trailing total included, so the counts must
// not read past an empty one.
TEST(BasisSet, MovedFromBasisHasNoFunctions) {
  fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(2, {{1.0, 1.0}}, kOrigin)});
  const fb::BasisSet taken = std::move(basis);
  EXPECT_EQ(taken.nao_cart(), 7U);
  EXPECT_EQ(taken.nao_sphe(), 6U);
  // NOLINTBEGIN(bugprone-use-after-move): the moved-from state is under test
  EXPECT_EQ(basis.nao_cart(), 0U);
  EXPECT_EQ(basis.nao_sphe(), 0U);
  // NOLINTEND(bugprone-use-after-move)
}

TEST(BasisSet, AtomShellOffsetsGroupShells) {
  const fb::BasisSet bare({fb::Shell(0, {{1.0, 1.0}}, kOrigin)});
  EXPECT_EQ(bare.natoms(), 0U);

  // Atom 0: s + p, atom 1: no shells, atom 2: s.
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                            fb::Shell(1, {{1.0, 1.0}}, kOrigin),
                            fb::Shell(0, {{1.0, 1.0}}, kOrigin)},
                           {0, 2, 2, 3});
  ASSERT_EQ(basis.natoms(), 3U);
  EXPECT_EQ(basis.atom_shell_offset(0), 0U);
  EXPECT_EQ(basis.atom_shell_count(0), 2U);
  EXPECT_EQ(basis.atom_shell_count(1), 0U);
  EXPECT_EQ(basis.atom_shell_offset(2), 2U);
  EXPECT_EQ(basis.atom_shell_count(2), 1U);
}

// Only place() from a compiled-in stamp names a basis; a shell list does not.
TEST(BasisSet, NameIsWhatTheConstructorWasGiven) {
  EXPECT_TRUE(fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin)}).name().empty());
  EXPECT_TRUE(fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin)}, {0, 1}).name().empty());
  const fb::BasisSet named({fb::Shell(0, {{1.0, 1.0}}, kOrigin)}, {0, 1}, "sto-3g");
  EXPECT_EQ(named.name(), "sto-3g");
  EXPECT_EQ(named.natoms(), 1U);
}

TEST(BasisSetDeathTest, AtomShellOffsetsAbortOnBadPartition) {
  const auto shells = [] {
    return std::vector<fb::Shell>{fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                                  fb::Shell(0, {{1.0, 1.0}}, kOrigin)};
  };
  EXPECT_DEATH(fb::BasisSet(shells(), {}), "");           // no trailing total
  EXPECT_DEATH(fb::BasisSet(shells(), {1, 2}), "");       // does not start at 0
  EXPECT_DEATH(fb::BasisSet(shells(), {0, 1}), "");       // does not end at nshells
  EXPECT_DEATH(fb::BasisSet(shells(), {0, 2, 1, 2}), ""); // decreases
  const fb::BasisSet basis(shells(), {0, 2});
  EXPECT_DEATH((void)basis.atom_shell_offset(1), "");
  EXPECT_DEATH((void)basis.atom_shell_count(1), "");
}

TEST(BasisSetDeathTest, AoOffsetAbortsOutOfRange) {
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin)});
  EXPECT_EQ(basis.ao_offset_cart(0), 0U);
  EXPECT_DEATH((void)basis.ao_offset_cart(1), "");
  EXPECT_DEATH((void)basis.ao_offset_sphe(1), "");
}

TEST(BasisSet, LayoutsConcatenateShellsInOrder) {
  // s, p, d.
  const fb::BasisSet basis({fb::Shell(0, {{3.42, 0.15}}, kOrigin),
                            fb::Shell(1, {{0.62, 1.0}}, kOrigin),
                            fb::Shell(2, {{0.8, 1.0}}, kOrigin)});
  EXPECT_EQ(basis.nshells(), 3U);
  EXPECT_EQ(basis.nao_cart(), 10U); // 1 + 3 + 6: the d shell's Cartesian functions
  EXPECT_EQ(basis.nao_sphe(), 9U);  // 1 + 3 + 5
  EXPECT_EQ(basis.ao_offset_cart(0), 0U);
  EXPECT_EQ(basis.ao_offset_cart(1), 1U);
  EXPECT_EQ(basis.ao_offset_cart(2), 4U);
  EXPECT_EQ(basis.ao_offset_sphe(2), 4U);

  const std::vector<fb::AO> &aos_cart = basis.aos_cart();
  ASSERT_EQ(aos_cart.size(), basis.nao_cart());
  for (std::uint32_t s = 0; s < basis.nshells(); ++s) {
    const std::vector<Powers> powers = fb::cartesian_powers(basis.shells()[s].angular_momentum());
    for (std::size_t i = 0; i < powers.size(); ++i) {
      const fb::AO &ao = aos_cart[basis.ao_offset_cart(s) + i];
      EXPECT_EQ(ao.shell_index, s);
      EXPECT_EQ((Powers{ao.a, ao.b, ao.c}), powers[i]);
    }
  }
}

TEST(BasisSet, SphericalTransformMatrixIsIdentityBelowD) {
  // s and p are the same functions in either convention.
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                            fb::Shell(1, {{1.0, 1.0}}, kOrigin),
                            fb::Shell(0, {{0.5, 1.0}}, kOrigin)});
  const std::size_t n = basis.nao_cart();
  ASSERT_EQ(basis.nao_sphe(), n);
  const std::vector<double> u = fb::spherical_transform_matrix(basis);
  ASSERT_EQ(u.size(), n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      EXPECT_EQ(u[i + (j * n)], i == j ? 1.0 : 0.0) << i << "," << j;
    }
  }
}

TEST(BasisSet, SphericalShellOverlapTransformsToIdentity) {
  // For one normalized primitive at a single center, the Cartesian overlap is
  // S_pq = c^2 (pi/2a)^{3/2} G(p,q) / (4a)^l in closed form. U^T S U must be
  // the identity on the spherical block: the property spherical_transform_matrix's
  // sqrt((2l-1)!!) scale exists to deliver.
  const double pi = std::numbers::pi;
  for (std::int32_t l = 2; l <= 4; ++l) {
    const double alpha = 1.3;
    const fb::BasisSet basis(
        {fb::place(fb::ShellStamp(l, {{alpha, 1.0}}).primitive_normalized(), kOrigin)});
    const double c = basis.shells()[0].primitives()[0].coefficient;
    const std::vector<Powers> powers = fb::cartesian_powers(l);
    const std::size_t ncart = basis.nao_cart();
    const std::size_t nsph = basis.nao_sphe();
    ASSERT_EQ(nsph, (2 * static_cast<std::size_t>(l)) + 1);

    const std::vector<double> u = fb::spherical_transform_matrix(basis);
    for (std::size_t m1 = 0; m1 < nsph; ++m1) {
      for (std::size_t m2 = 0; m2 < nsph; ++m2) {
        double s = 0.0;
        for (std::size_t p = 0; p < ncart; ++p) {
          for (std::size_t q = 0; q < ncart; ++q) {
            const double s_pq = c * c * std::pow(pi / (2.0 * alpha), 1.5) *
                                monomial_gram(powers[p], powers[q]) / std::pow(4.0 * alpha, l);
            s += u[p + (m1 * ncart)] * s_pq * u[q + (m2 * ncart)];
          }
        }
        EXPECT_NEAR(s, m1 == m2 ? 1.0 : 0.0, 1e-12) << "l=" << l << " m1=" << m1 << " m2=" << m2;
      }
    }
  }
}

TEST(BasisSet, SphericalTransformMatrixPlacesBlocksAtOffsets) {
  // s then d: U is 7 x 6, block-diagonal, column-major.
  const fb::BasisSet basis(
      {fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(2, {{1.0, 1.0}}, kOrigin)});
  const std::size_t ncol = basis.nao_sphe();
  ASSERT_EQ(basis.nao_cart(), 7U);
  ASSERT_EQ(ncol, 6U);
  const std::vector<double> u = fb::spherical_transform_matrix(basis);
  const std::vector<double> d = fb::solid_harmonic_block(2);
  EXPECT_EQ(u[0], 1.0);
  for (std::size_t j = 1; j < ncol; ++j) {
    EXPECT_EQ(u[j * 7], 0.0); // s row, d columns
  }
  for (std::size_t i = 1; i < 7; ++i) {
    EXPECT_EQ(u[i], 0.0); // d rows, s column
    for (std::size_t j = 0; j < 5; ++j) {
      EXPECT_NEAR(u[i + ((1 + j) * 7)], std::sqrt(3.0) * d[(i - 1) + (j * 6)], 1e-15);
    }
  }
}

// U's columns are orthonormal under the Cartesian overlap, not the plain dot
// product, so U^T U != I and spherical -> Cartesian -> spherical is no round
// trip. The d_xy column (m = -2) is sqrt(3) in the xy row and zero elsewhere,
// so its U^T U diagonal is 3.
TEST(BasisSet, SphericalTransformMatrixIsNotOrthonormal) {
  const fb::BasisSet basis({fb::Shell(2, {{1.0, 1.0}}, kOrigin)});
  const std::size_t ncart = basis.nao_cart();
  const std::size_t nsph = basis.nao_sphe();
  const std::vector<double> u = fb::spherical_transform_matrix(basis);
  ASSERT_EQ(u.size(), ncart * nsph);

  std::vector<double> utu(nsph * nsph, 0.0);
  for (std::size_t a = 0; a < nsph; ++a) {
    for (std::size_t b = 0; b < nsph; ++b) {
      for (std::size_t i = 0; i < ncart; ++i) {
        utu[a + (b * nsph)] += u[i + (a * ncart)] * u[i + (b * ncart)];
      }
    }
  }
  EXPECT_NEAR(u[row_of(2, {1, 1, 0})], std::sqrt(3.0), 1e-15);
  EXPECT_NEAR(utu[0], 3.0, 1e-14);
  double off_identity = 0.0;
  for (std::size_t a = 0; a < nsph; ++a) {
    for (std::size_t b = 0; b < nsph; ++b) {
      off_identity = std::max(off_identity, std::abs(utu[a + (b * nsph)] - (a == b ? 1.0 : 0.0)));
    }
  }
  EXPECT_GT(off_identity, 1.0);
}

} // namespace
