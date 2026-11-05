// Host cases for recipes.matrix_ao's ao_count<L>: the order a MatrixAO<L>
// takes from a basis, against hand counts. A Cartesian shell of angular
// momentum l has (l+1)(l+2)/2 functions and a spherical one 2l+1, so s and p
// count the same in both layouts and d (6 against 5) and f (10 against 7)
// are where they part. ao_count reads only the host BasisSet, so no case
// needs a device. The counts are integers and are checked exactly.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

fb::Shell shell(const std::int32_t l) {
  return fb::Shell(l, {{1.0, 1.0}}, kOrigin);
}

TEST(MatrixAOCount, EmptyBasisHasNoFunctions) {
  const fb::BasisSet basis;
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Cartesian>(basis), 0U);
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Spherical>(basis), 0U);
}

TEST(MatrixAOCount, LayoutsAgreeThroughP) {
  // s + p + p: 1 + 3 + 3 in both layouts.
  const fb::BasisSet basis({shell(0), shell(1), shell(1)});
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Cartesian>(basis), 7U);
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Spherical>(basis), 7U);
}

TEST(MatrixAOCount, OneShellOfEachMomentum) {
  // Each l alone: Cartesian (l+1)(l+2)/2, spherical 2l+1.
  constexpr std::array<std::uint32_t, 4> kCartesian{1, 3, 6, 10};
  constexpr std::array<std::uint32_t, 4> kSpherical{1, 3, 5, 7};
  for (std::int32_t l = 0; l < 4; ++l) {
    const fb::BasisSet basis({shell(l)});
    const auto at = static_cast<std::size_t>(l);
    EXPECT_EQ(fb::ao_count<fb::BasisLayout::Cartesian>(basis), kCartesian[at]) << "l = " << l;
    EXPECT_EQ(fb::ao_count<fb::BasisLayout::Spherical>(basis), kSpherical[at]) << "l = " << l;
  }
}

TEST(MatrixAOCount, MixedBasisThroughF) {
  // s + p + d + f + s + d, a shell of each l and a repeat of two:
  // Cartesian 1 + 3 + 6 + 10 + 1 + 6 = 27, spherical 1 + 3 + 5 + 7 + 1 + 5 = 22.
  const fb::BasisSet basis({shell(0), shell(1), shell(2), shell(3), shell(0), shell(2)});
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Cartesian>(basis), 27U);
  EXPECT_EQ(fb::ao_count<fb::BasisLayout::Spherical>(basis), 22U);
}

} // namespace
