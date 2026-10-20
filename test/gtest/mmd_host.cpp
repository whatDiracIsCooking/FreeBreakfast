// Host cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field against the Gaussian product theorem, the la <= lb
// swap that lets one kernel serve both (s|p) and (p|s), and the refusal of
// storage that does not fit the basis.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

namespace {

namespace fb = free_breakfast;

constexpr double kTol = 1e-15;

// A basis's records in host memory, the storage owned here as a caller of
// write_primitive_pairs would own it.
struct HostRecords {
  std::vector<fb::PrimitivePairClass> classes;
  std::vector<std::vector<double>> reals;
  std::vector<std::vector<std::int32_t>> offsets;

  double real(const std::size_t c, const fb::PairReal f, const std::size_t k) const {
    return reals[c][fb::field_offset(f, classes[c].n_rec) + k];
  }
  std::int32_t offset(const std::size_t c, const fb::PairOffset f, const std::size_t k) const {
    return offsets[c][fb::field_offset(f, classes[c].n_rec) + k];
  }
};

HostRecords host_records(const fb::BasisSet &basis) {
  HostRecords h{.classes = fb::primitive_pair_classes(basis)};
  std::vector<fb::PrimitivePairRecords> records;
  for (const fb::PrimitivePairClass &cls : h.classes) {
    h.reals.emplace_back(static_cast<std::size_t>(fb::kPairReals * cls.n_rec));
    h.offsets.emplace_back(static_cast<std::size_t>(fb::kPairOffsets * cls.n_rec));
  }
  for (std::size_t c = 0; c < h.classes.size(); ++c) {
    records.push_back({.reals = h.reals[c], .offsets = h.offsets[c]});
  }
  fb::write_primitive_pairs(basis, h.classes, records);
  return h;
}

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
  EXPECT_EQ(classes[0].n_rec, 1);
  EXPECT_EQ(classes[1].n_rec, 2);
  EXPECT_EQ(classes[2].n_rec, 4);
}

TEST(Mmd, FieldsAreLaidOutOneBlockOfNRecEach) {
  EXPECT_EQ(fb::field_offset(fb::PairReal::Zeta, 5), 0U);
  EXPECT_EQ(fb::field_offset(fb::PairReal::PX, 5), 5U);
  EXPECT_EQ(fb::field_offset(fb::PairReal::Eb, 5), 55U);
  EXPECT_EQ(static_cast<std::int32_t>(fb::PairReal::Eb) + 1, fb::kPairReals);
  EXPECT_EQ(fb::field_offset(fb::PairOffset::A, 5), 0U);
  EXPECT_EQ(fb::field_offset(fb::PairOffset::B, 5), 5U);
  EXPECT_EQ(static_cast<std::int32_t>(fb::PairOffset::B) + 1, fb::kPairOffsets);
}

TEST(Mmd, HigherMomentumBraIsSwappedToTheKet) {
  using R = fb::PairReal;
  const fb::BasisSet basis = p_then_s();
  const HostRecords h = host_records(basis);
  constexpr std::size_t kSp = 1;
  ASSERT_EQ(h.classes[kSp].la, 0);
  ASSERT_EQ(h.classes[kSp].lb, 1);

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
    EXPECT_EQ(h.offset(kSp, fb::PairOffset::A, k), s_off) << "record " << k;
    EXPECT_EQ(h.offset(kSp, fb::PairOffset::B, k), p_off) << "record " << k;
    EXPECT_NEAR(h.real(kSp, R::Zeta, k), zeta, kTol) << "record " << k;
    EXPECT_NEAR(h.real(kSp, R::Eb, k), beta, kTol) << "record " << k;
    EXPECT_NEAR(h.real(kSp, R::PZ, k), pz, kTol) << "record " << k;
    EXPECT_NEAR(h.real(kSp, R::PaZ, k), pz - z_a, kTol) << "record " << k;
    EXPECT_NEAR(h.real(kSp, R::PbZ, k), pz - z_b, kTol) << "record " << k;
    // x and y: every center is on the z axis
    for (const R axis : {R::PX, R::PY, R::PaX, R::PaY, R::PbX, R::PbY}) {
      EXPECT_EQ(h.real(kSp, axis, k), 0.0) << "record " << k;
    }
    const double s = c_a * ket[k].coefficient * std::pow(std::numbers::pi / zeta, 1.5);
    EXPECT_NEAR(h.real(kSp, R::S, k), s, kTol * s) << "record " << k;
  }
}

TEST(Mmd, DiagonalShellPairCarriesEqualOffsets) {
  const HostRecords h = host_records(p_then_s());
  for (const std::size_t c : {0U, 2U}) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(h.classes[c].n_rec); ++k) {
      EXPECT_EQ(h.offset(c, fb::PairOffset::A, k), h.offset(c, fb::PairOffset::B, k))
          << "(" << h.classes[c].la << ", " << h.classes[c].lb << ")";
    }
  }
  // Records are bra-primitive-major: (alpha_0, alpha_0), (alpha_0, alpha_1), ...
  constexpr std::size_t kPp = 2;
  EXPECT_NEAR(h.real(kPp, fb::PairReal::Zeta, 0), 4.0, kTol);
  EXPECT_NEAR(h.real(kPp, fb::PairReal::Zeta, 1), 2.5, kTol);
  EXPECT_NEAR(h.real(kPp, fb::PairReal::Zeta, 2), 2.5, kTol);
  EXPECT_NEAR(h.real(kPp, fb::PairReal::Zeta, 3), 1.0, kTol);
}

TEST(Mmd, EmptyBasisHasNoClasses) {
  const fb::BasisSet empty{};
  EXPECT_TRUE(fb::primitive_pair_classes(empty).empty());
  fb::write_primitive_pairs(empty, {}, {}); // nothing to write, nothing refused
}

// write_primitive_pairs writes by index into the caller's spans, so storage
// that does not fit the basis is refused before anything is written.
TEST(MmdDeathTest, WriteRefusesMismatchedStorage) {
  const fb::BasisSet basis = p_then_s();
  const auto classes = fb::primitive_pair_classes(basis);
  std::vector<double> reals(static_cast<std::size_t>(fb::kPairReals * classes[0].n_rec));
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(fb::kPairOffsets * classes[0].n_rec));
  const std::vector<fb::PrimitivePairRecords> one{{.reals = reals, .offsets = offsets}};
  // One record set for three classes.
  EXPECT_DEATH(fb::write_primitive_pairs(basis, classes, one), "");
  // The right count, but class 1 sized for class 0.
  const std::vector<fb::PrimitivePairRecords> short_sets(classes.size(), one[0]);
  EXPECT_DEATH(fb::write_primitive_pairs(basis, classes, short_sets), "");
  // Classes that are not this basis's.
  EXPECT_DEATH(fb::write_primitive_pairs(basis, std::span{classes}.first(1), one), "");
}

} // namespace
