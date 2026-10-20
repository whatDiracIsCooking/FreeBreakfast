// Host cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field against the Gaussian product theorem, the la <= lb
// swap that lets one kernel serve both (s|p) and (p|s), the PairFields walkers,
// and the refusal of storage that does not fit the basis.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

namespace {

namespace fb = free_breakfast;

constexpr double kTol = 1e-15;

// One class's records in host memory, owned here as a caller of
// write_primitive_pairs would own them.
using HostFields = fb::PairFields<std::vector<double>, std::vector<std::int32_t>>;

// Every field n long.
HostFields host_fields(const std::size_t n) {
  return fb::map_pair_fields([n]<typename T>(const T &) { return std::vector<T>(n); },
                             fb::PairFields<double, std::int32_t>{});
}

// Spans over every field of h, for write_primitive_pairs.
fb::PrimitivePairRecords spans_of(HostFields &h) {
  return fb::map_pair_fields([](auto &v) { return std::span{v}; }, h);
}

struct HostRecords {
  std::vector<fb::PrimitivePairClass> classes;
  std::vector<HostFields> fields;
};

HostRecords host_records(const fb::BasisSet &basis) {
  HostRecords h{.classes = fb::primitive_pair_classes(basis)};
  for (const fb::PrimitivePairClass &cls : h.classes) {
    h.fields.push_back(host_fields(static_cast<std::size_t>(cls.n_rec)));
  }
  std::vector<fb::PrimitivePairRecords> records;
  for (HostFields &f : h.fields) {
    records.push_back(spans_of(f));
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

// apply_pair_fields visits every field once, in declaration order, and
// map_pair_fields keeps each value in its own field.
TEST(Mmd, PairFieldsWalkersVisitEveryFieldInOrder) {
  fb::PairFields<std::int32_t, std::int32_t> f{};
  std::int32_t next = 0;
  fb::apply_pair_fields([&next](auto &...field) { ((field = next++), ...); }, f);
  EXPECT_EQ(static_cast<std::size_t>(next), fb::kPairFields);
  EXPECT_EQ(f.zeta, 0);
  EXPECT_EQ(f.p_x, 1);
  EXPECT_EQ(f.pb_z, 9);
  EXPECT_EQ(f.eb, 11);
  EXPECT_EQ(f.off_a, 12);
  EXPECT_EQ(f.off_b, 13);

  const auto doubled = fb::map_pair_fields([](const std::int32_t v) { return 2.0 * v; }, f);
  static_assert(std::same_as<decltype(doubled), const fb::PairFields<double, double>>);
  EXPECT_EQ(doubled.zeta, 0.0);
  EXPECT_EQ(doubled.s, 20.0);
  EXPECT_EQ(doubled.off_b, 26.0);
}

TEST(Mmd, HigherMomentumBraIsSwappedToTheKet) {
  const fb::BasisSet basis = p_then_s();
  const HostRecords h = host_records(basis);
  ASSERT_EQ(h.classes[1].la, 0);
  ASSERT_EQ(h.classes[1].lb, 1);
  const HostFields &sp = h.fields[1];

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
    EXPECT_NEAR(sp.p_z[k], pz, kTol) << "record " << k;
    EXPECT_NEAR(sp.pa_z[k], pz - z_a, kTol) << "record " << k;
    EXPECT_NEAR(sp.pb_z[k], pz - z_b, kTol) << "record " << k;
    // x and y: every center is on the z axis
    for (const std::vector<double> *axis :
         {&sp.p_x, &sp.p_y, &sp.pa_x, &sp.pa_y, &sp.pb_x, &sp.pb_y}) {
      EXPECT_EQ((*axis)[k], 0.0) << "record " << k;
    }
    const double s = c_a * ket[k].coefficient * std::pow(std::numbers::pi / zeta, 1.5);
    EXPECT_NEAR(sp.s[k], s, kTol * s) << "record " << k;
  }
}

TEST(Mmd, DiagonalShellPairCarriesEqualOffsets) {
  const HostRecords h = host_records(p_then_s());
  for (const std::size_t c : {0U, 2U}) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(h.classes[c].n_rec); ++k) {
      EXPECT_EQ(h.fields[c].off_a[k], h.fields[c].off_b[k])
          << "(" << h.classes[c].la << ", " << h.classes[c].lb << ")";
    }
  }
  // Records are bra-primitive-major: (alpha_0, alpha_0), (alpha_0, alpha_1), ...
  const HostFields &pp = h.fields[2];
  EXPECT_NEAR(pp.zeta[0], 4.0, kTol);
  EXPECT_NEAR(pp.zeta[1], 2.5, kTol);
  EXPECT_NEAR(pp.zeta[2], 2.5, kTol);
  EXPECT_NEAR(pp.zeta[3], 1.0, kTol);
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
  HostFields first = host_fields(static_cast<std::size_t>(classes[0].n_rec));
  const std::vector<fb::PrimitivePairRecords> one{spans_of(first)};
  // One record set for three classes.
  EXPECT_DEATH(fb::write_primitive_pairs(basis, classes, one), "");
  // The right count, but every class sized for class 0.
  const std::vector<fb::PrimitivePairRecords> short_sets(classes.size(), one[0]);
  EXPECT_DEATH(fb::write_primitive_pairs(basis, classes, short_sets), "");
  // Classes that are not this basis's.
  EXPECT_DEATH(fb::write_primitive_pairs(basis, std::span{classes}.first(1), one), "");
  // The right sizes but for one field.
  std::vector<HostFields> fields;
  for (const fb::PrimitivePairClass &cls : classes) {
    fields.push_back(host_fields(static_cast<std::size_t>(cls.n_rec)));
  }
  fields[2].eb.pop_back();
  std::vector<fb::PrimitivePairRecords> one_short;
  for (HostFields &f : fields) {
    one_short.push_back(spans_of(f));
  }
  EXPECT_DEATH(fb::write_primitive_pairs(basis, classes, one_short), "");
}

} // namespace
