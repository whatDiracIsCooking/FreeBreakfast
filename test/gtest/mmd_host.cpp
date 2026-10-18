// Host cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field against the Gaussian product theorem, and the
// la <= lb swap that lets one kernel serve both (s|p) and (p|s).

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

namespace {

namespace fb = free_breakfast;

constexpr double kTol = 1e-15;

// A two-primitive p shell on A, then a one-primitive s shell on B: the p
// precedes the s in basis order, so the (p, s) shell pair must come out as an
// (s|p) record with the s shell as the bra.
fb::BasisSet p_then_s() {
  const fb::Vec<double, 3> a{0.0, 0.0, 0.0};
  const fb::Vec<double, 3> b{0.0, 0.0, 1.5};
  return fb::BasisSet({
      fb::Shell(1, {{2.0, 0.4}, {0.5, 0.7}}, a),
      fb::Shell(0, {{1.2, 0.9}}, b),
  });
}

TEST(Mmd, ClassesAreReachedOnlyAndAscending) {
  const auto classes = fb::primitive_pair_classes(p_then_s());
  ASSERT_EQ(classes.size(), 3U);
  const std::vector<std::pair<std::int32_t, std::int32_t>> want{{0, 0}, {0, 1}, {1, 1}};
  for (std::size_t c = 0; c < want.size(); ++c) {
    EXPECT_EQ(classes[c].la, want[c].first) << "class " << c;
    EXPECT_EQ(classes[c].lb, want[c].second) << "class " << c;
  }
  // (s|s): 1 x 1 primitives; (s|p): 1 x 2; (p|p): 2 x 2.
  EXPECT_EQ(classes[0].size(), 1U);
  EXPECT_EQ(classes[1].size(), 2U);
  EXPECT_EQ(classes[2].size(), 4U);
}

TEST(Mmd, HigherMomentumBraIsSwappedToTheKet) {
  const fb::BasisSet basis = p_then_s();
  const auto classes = fb::primitive_pair_classes(basis);
  const fb::PrimitivePairClass &sp = classes[1];
  ASSERT_EQ(sp.la, 0);
  ASSERT_EQ(sp.lb, 1);

  // The bra is the s shell (shell 1, AO 3), the ket the p shell (AOs 0..2).
  const std::int32_t s_off = static_cast<std::int32_t>(basis.ao_offset_cart(1));
  const std::int32_t p_off = static_cast<std::int32_t>(basis.ao_offset_cart(0));
  ASSERT_EQ(s_off, 3);
  ASSERT_EQ(p_off, 0);

  const double z_a = 1.5; // the s shell's center: the bra's, after the swap
  const double z_b = 0.0;
  const double alpha = 1.2;
  const double c_a = 0.9;
  const std::vector<fb::Primitive> ket = basis.shells()[0].primitives();
  for (std::size_t k = 0; k < ket.size(); ++k) {
    const double beta = ket[k].exponent;
    const double zeta = alpha + beta;
    const double pz = (alpha * z_a + beta * z_b) / zeta;
    EXPECT_EQ(sp.off_a[k], s_off) << "record " << k;
    EXPECT_EQ(sp.off_b[k], p_off) << "record " << k;
    EXPECT_NEAR(sp.zeta[k], zeta, kTol) << "record " << k;
    EXPECT_NEAR(sp.eb[k], beta, kTol) << "record " << k;
    EXPECT_NEAR(sp.p[3 * k + 2], pz, kTol) << "record " << k;
    EXPECT_NEAR(sp.xpa[3 * k + 2], pz - z_a, kTol) << "record " << k;
    EXPECT_NEAR(sp.xpb[3 * k + 2], pz - z_b, kTol) << "record " << k;
    for (std::size_t d = 0; d < 2; ++d) { // x and y: every center is on the z axis
      EXPECT_EQ(sp.p[3 * k + d], 0.0);
      EXPECT_EQ(sp.xpa[3 * k + d], 0.0);
      EXPECT_EQ(sp.xpb[3 * k + d], 0.0);
    }
    const double s = c_a * ket[k].coefficient * std::pow(std::numbers::pi / zeta, 1.5);
    EXPECT_NEAR(sp.s[k], s, kTol * s) << "record " << k;
  }
}

TEST(Mmd, DiagonalShellPairCarriesEqualOffsets) {
  const auto classes = fb::primitive_pair_classes(p_then_s());
  for (const fb::PrimitivePairClass *cls : {&classes[0], &classes[2]}) {
    for (std::size_t k = 0; k < cls->size(); ++k) {
      EXPECT_EQ(cls->off_a[k], cls->off_b[k]) << "(" << cls->la << ", " << cls->lb << ")";
    }
  }
  // Records are bra-primitive-major: (alpha_0, alpha_0), (alpha_0, alpha_1), ...
  const fb::PrimitivePairClass &pp = classes[2];
  EXPECT_NEAR(pp.zeta[0], 4.0, kTol);
  EXPECT_NEAR(pp.zeta[1], 2.5, kTol);
  EXPECT_NEAR(pp.zeta[2], 2.5, kTol);
  EXPECT_NEAR(pp.zeta[3], 1.0, kTol);
}

TEST(Mmd, EmptyBasisHasNoClasses) {
  const auto classes = fb::primitive_pair_classes(fb::BasisSet{});
  EXPECT_TRUE(classes.empty());
}

} // namespace
