// Host-only cases for basis's :harmonics: the closed-form real solid-harmonic
// table, checked for orthonormality and against libcint's coefficients.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

#include "gaussian_moments.h"

namespace {

namespace fb = free_breakfast;
using fb::test::monomial_gram;
using fb::test::Powers;
using fb::test::row_of;

TEST(BasisHarmonics, SolidHarmonicsBelowDAreIdentity) {
  EXPECT_EQ(fb::solid_harmonic_block(0), (std::vector<double>{1.0}));
  EXPECT_EQ(fb::solid_harmonic_block(1),
            (std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
}

TEST(BasisHarmonics, SolidHarmonicsAreGOrthonormal) {
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

TEST(BasisHarmonics, SolidHarmonicDMatchesLibcintUpToOneScale) {
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

TEST(BasisHarmonics, SolidHarmonicFMatchesLibcintUpToOneScale) {
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

TEST(BasisHarmonics, SolidHarmonicFFollowsRealHarmonicConvention) {
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

} // namespace
