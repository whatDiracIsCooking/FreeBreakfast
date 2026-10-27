// Host-only cases for basis: the Cartesian ordering, the AO layouts, the
// closed-form solid-harmonic table and the Cartesian-to-spherical transform
// built from it, and primitive normalization. Placing a stamp onto a molecule
// is basis_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

namespace {

namespace fb = free_breakfast;
using Powers = std::array<std::int32_t, 3>;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

// (n)!! over odd n, the empty product for n <= 1.
double double_factorial(std::int32_t n) {
  double result = 1.0;
  for (std::int32_t k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

// The angular Gaussian-moment inner product the harmonic table is normalized
// under: prod over axes of (p+q-1)!!, zero if any axis sum is odd.
double monomial_gram(const Powers &p, const Powers &q) {
  double result = 1.0;
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    const std::int32_t s = p[axis] + q[axis];
    if (s % 2 != 0) {
      return 0.0;
    }
    result *= double_factorial(s - 1);
  }
  return result;
}

// Row index of the Cartesian monomial `p` within its shell.
std::size_t row_of(std::int32_t l, const Powers &p) {
  const std::vector<Powers> powers = fb::cartesian_powers(l);
  const auto it = std::ranges::find(powers, p);
  return static_cast<std::size_t>(it - powers.begin());
}

TEST(Basis, FunctionCounts) {
  EXPECT_EQ(fb::kMaxAngularMomentum, 3);
  const std::array<std::uint32_t, 5> cart{1, 3, 6, 10, 15};
  for (std::int32_t l = 0; l < 5; ++l) {
    EXPECT_EQ(fb::function_count_cart(l), cart[l]) << "l=" << l;
    EXPECT_EQ(fb::function_count_sphe(l), static_cast<std::uint32_t>((2 * l) + 1));
  }
}

TEST(Basis, CartesianPowersCanonicalOrder) {
  EXPECT_EQ(fb::cartesian_powers(0), (std::vector<Powers>{{0, 0, 0}}));
  EXPECT_EQ(fb::cartesian_powers(1), (std::vector<Powers>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
  // xx, xy, xz, yy, yz, zz
  EXPECT_EQ(
      fb::cartesian_powers(2),
      (std::vector<Powers>{{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}}));
  // Defined past d, with every entry of degree l.
  for (std::int32_t l = 3; l <= 5; ++l) {
    const std::vector<Powers> powers = fb::cartesian_powers(l);
    ASSERT_EQ(powers.size(), fb::function_count_cart(l));
    EXPECT_EQ(powers.front(), (Powers{l, 0, 0}));
    EXPECT_EQ(powers.back(), (Powers{0, 0, l}));
    for (const Powers &p : powers) {
      EXPECT_EQ(p[0] + p[1] + p[2], l);
    }
  }
}

TEST(Basis, ShellZipsParallelArrays) {
  const fb::Shell shell(2, {3.0, 0.5}, {0.25, 0.75}, fb::Vec<double, 3>{1.0, 2.0, 3.0});
  EXPECT_EQ(shell.angular_momentum(), 2);
  ASSERT_EQ(shell.nprimitives(), 2U);
  EXPECT_EQ(shell.primitives()[1].exponent, 0.5);
  EXPECT_EQ(shell.primitives()[1].coefficient, 0.75);
  EXPECT_EQ(shell.center().z, 3.0);
  EXPECT_EQ(shell.function_count_cart(), 6U); // Cartesian, what the kernels evaluate
  EXPECT_EQ(shell.function_count_sphe(), 5U); // 2l+1, what the shell means
}

TEST(BasisDeathTest, ShellAbortsOnBadShape) {
  EXPECT_DEATH(fb::Shell(-1, {{1.0, 1.0}}, kOrigin), "");
  EXPECT_DEATH(fb::Shell(0, std::vector<fb::Primitive>{}, kOrigin), "");
  EXPECT_DEATH(fb::Shell(0, std::vector<double>{1.0, 2.0}, std::vector<double>{1.0}, kOrigin), "");
  EXPECT_DEATH(fb::ShellStamp(-1, {{1.0, 1.0}}), "");
}

TEST(Basis, EmptyBasisHasNoFunctions) {
  const fb::BasisSet basis;
  EXPECT_EQ(basis.nshells(), 0U);
  EXPECT_EQ(basis.nao_cart(), 0U);
  EXPECT_EQ(basis.nao_sphe(), 0U);
  EXPECT_TRUE(basis.aos_cart().empty());
  EXPECT_TRUE(fb::spherical_transform_matrix(basis).empty());
}

// Moving takes the offset vectors, trailing total included, so the counts must
// not read past an empty one.
TEST(Basis, MovedFromBasisHasNoFunctions) {
  fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(2, {{1.0, 1.0}}, kOrigin)});
  const fb::BasisSet taken = std::move(basis);
  EXPECT_EQ(taken.nao_cart(), 7U);
  EXPECT_EQ(taken.nao_sphe(), 6U);
  // NOLINTBEGIN(bugprone-use-after-move): the moved-from state is under test
  EXPECT_EQ(basis.nao_cart(), 0U);
  EXPECT_EQ(basis.nao_sphe(), 0U);
  // NOLINTEND(bugprone-use-after-move)
}

TEST(Basis, AtomShellOffsetsGroupShells) {
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

TEST(BasisDeathTest, AtomShellOffsetsAbortOnBadPartition) {
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

TEST(BasisDeathTest, AoOffsetAbortsOutOfRange) {
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, kOrigin)});
  EXPECT_EQ(basis.ao_offset_cart(0), 0U);
  EXPECT_DEATH((void)basis.ao_offset_cart(1), "");
  EXPECT_DEATH((void)basis.ao_offset_sphe(1), "");
}

TEST(Basis, LayoutsConcatenateShellsInOrder) {
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

TEST(Basis, SphericalTransformMatrixIsIdentityBelowD) {
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

TEST(Basis, SolidHarmonicsBelowDAreIdentity) {
  EXPECT_EQ(fb::solid_harmonic_block(0), (std::vector<double>{1.0}));
  EXPECT_EQ(fb::solid_harmonic_block(1),
            (std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
}

TEST(Basis, SolidHarmonicsAreGOrthonormal) {
  for (std::int32_t l = 0; l <= 5; ++l) {
    const std::vector<Powers> powers = fb::cartesian_powers(l);
    const std::size_t ncart = powers.size();
    const std::size_t nsph = (2 * static_cast<std::size_t>(l)) + 1;
    const std::vector<double> b = fb::solid_harmonic_block(l);
    ASSERT_EQ(b.size(), ncart * nsph);
    for (std::size_t m1 = 0; m1 < nsph; ++m1) {
      for (std::size_t m2 = 0; m2 < nsph; ++m2) {
        double dot = 0.0;
        for (std::size_t p = 0; p < ncart; ++p) {
          for (std::size_t q = 0; q < ncart; ++q) {
            dot += b[p + (m1 * ncart)] * monomial_gram(powers[p], powers[q]) * b[q + (m2 * ncart)];
          }
        }
        EXPECT_NEAR(dot, m1 == m2 ? 1.0 : 0.0, 1e-12) << "l=" << l << " m1=" << m1 << " m2=" << m2;
      }
    }
  }
}

// Our block for momentum l against libcint's (PySCF's) cart2sph block, given
// row by row (rows in cartesian_powers order, columns m = -l..+l). The two
// differ only in the overall normalization, so one positive scalar, read off
// the entry at (`row`, `col`), maps every column of libcint's onto ours --
// same order, same phases.
void expect_matches_libcint_up_to_one_scale(std::int32_t l, std::span<const double> libcint,
                                            const Powers &row, std::size_t col) {
  const std::vector<double> block = fb::solid_harmonic_block(l);
  const std::size_t ncart = fb::function_count_cart(l);
  const std::size_t nsph = fb::function_count_sphe(l);
  ASSERT_EQ(block.size(), libcint.size());
  const double scale =
      block[row_of(l, row) + (col * ncart)] / libcint[(row_of(l, row) * nsph) + col];
  EXPECT_GT(scale, 0.0);
  for (std::size_t r = 0; r < ncart; ++r) {
    for (std::size_t c = 0; c < nsph; ++c) {
      EXPECT_NEAR(block[r + (c * ncart)], scale * libcint[(r * nsph) + c], 1e-14)
          << "l=" << l << " " << r << "," << c;
    }
  }
}

TEST(Basis, SolidHarmonicDMatchesLibcintUpToOneScale) {
  // libcint's d-shell cart2sph block, columns m = -2..+2
  // (dxy, dyz, dz2, dxz, dx2-y2), in closed form.
  const double pi = std::numbers::pi;
  const double a = std::sqrt(15.0 / (4.0 * pi));  // 1.0925484305920792
  const double b = std::sqrt(5.0 / (16.0 * pi));  // 0.31539156525252005
  const double c = std::sqrt(15.0 / (16.0 * pi)); // 0.5462742152960396
  // Rows xx, xy, xz, yy, yz, zz.
  const std::array<double, 30> libcint{
      0.0, 0.0, -b,      0.0, c,   //
      a,   0.0, 0.0,     0.0, 0.0, //
      0.0, 0.0, 0.0,     a,   0.0, //
      0.0, 0.0, -b,      0.0, -c,  //
      0.0, a,   0.0,     0.0, 0.0, //
      0.0, 0.0, 2.0 * b, 0.0, 0.0, //
  };
  expect_matches_libcint_up_to_one_scale(2, libcint, {1, 1, 0}, 0);
}

TEST(Basis, SolidHarmonicFMatchesLibcintUpToOneScale) {
  // libcint's f-shell cart2sph block, columns m = -3..+3, in closed form.
  const double pi = std::numbers::pi;
  const double a = std::sqrt(35.0 / (32.0 * pi));  // 0.5900435899266435
  const double b = std::sqrt(105.0 / (4.0 * pi));  // 2.890611442640554
  const double c = std::sqrt(21.0 / (32.0 * pi));  // 0.4570457994644658
  const double d = std::sqrt(7.0 / (16.0 * pi));   // 0.3731763325901154
  const double e = std::sqrt(105.0 / (16.0 * pi)); // 1.445305721320277
  // Rows xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
  const std::array<double, 70> libcint{
      0.0,     0.0, 0.0,     0.0,      -c,      0.0, a,        //
      3.0 * a, 0.0, -c,      0.0,      0.0,     0.0, 0.0,      //
      0.0,     0.0, 0.0,     -3.0 * d, 0.0,     e,   0.0,      //
      0.0,     0.0, 0.0,     0.0,      -c,      0.0, -3.0 * a, //
      0.0,     b,   0.0,     0.0,      0.0,     0.0, 0.0,      //
      0.0,     0.0, 0.0,     0.0,      4.0 * c, 0.0, 0.0,      //
      -a,      0.0, -c,      0.0,      0.0,     0.0, 0.0,      //
      0.0,     0.0, 0.0,     -3.0 * d, 0.0,     -e,  0.0,      //
      0.0,     0.0, 4.0 * c, 0.0,      0.0,     0.0, 0.0,      //
      0.0,     0.0, 0.0,     2.0 * d,  0.0,     0.0, 0.0,      //
  };
  expect_matches_libcint_up_to_one_scale(3, libcint, {1, 1, 1}, 1);
}

TEST(Basis, SolidHarmonicFFollowsRealHarmonicConvention) {
  const std::vector<double> block = fb::solid_harmonic_block(3);
  auto at = [&](const Powers &p, std::int32_t m) { return block[row_of(3, p) + ((m + 3) * 10)]; };
  // m = -3: 3x^2 y - y^3
  EXPECT_GT(at({2, 1, 0}, -3), 0.0);
  EXPECT_NEAR(at({0, 3, 0}, -3) / at({2, 1, 0}, -3), -1.0 / 3.0, 1e-14);
  // m = 0: 2z^3 - 3x^2 z - 3y^2 z
  EXPECT_GT(at({0, 0, 3}, 0), 0.0);
  EXPECT_NEAR(at({2, 0, 1}, 0) / at({0, 0, 3}, 0), -1.5, 1e-14);
  EXPECT_NEAR(at({0, 2, 1}, 0) / at({0, 0, 3}, 0), -1.5, 1e-14);
  EXPECT_EQ(at({1, 1, 1}, 0), 0.0);
  // m = +3: x^3 - 3x y^2
  EXPECT_GT(at({3, 0, 0}, 3), 0.0);
  EXPECT_NEAR(at({1, 2, 0}, 3) / at({3, 0, 0}, 3), -3.0, 1e-14);
}

TEST(Basis, ShellStampNormalizesEachPrimitive) {
  const double pi = std::numbers::pi;
  for (std::int32_t l = 0; l <= 3; ++l) {
    const fb::ShellStamp stamp =
        fb::ShellStamp(l, std::vector<double>{2.0, 0.3}, std::vector<double>{0.4, 0.6})
            .primitive_normalized();
    for (const auto &[i, alpha, c] : {std::tuple{0, 2.0, 0.4}, std::tuple{1, 0.3, 0.6}}) {
      const double norm = std::pow(2.0 * alpha / pi, 0.75) * std::pow(4.0 * alpha, l / 2.0) /
                          std::sqrt(double_factorial((2 * l) - 1));
      EXPECT_NEAR(stamp.primitives()[i].coefficient, c * norm, 1e-12) << "l=" << l;
      EXPECT_EQ(stamp.primitives()[i].exponent, alpha);
    }
  }
}

TEST(Basis, SphericalShellOverlapTransformsToIdentity) {
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

TEST(Basis, SphericalTransformMatrixPlacesBlocksAtOffsets) {
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
TEST(Basis, SphericalTransformMatrixIsNotOrthonormal) {
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
