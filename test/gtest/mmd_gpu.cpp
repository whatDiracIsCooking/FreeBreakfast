// GPU cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field (on each class's pinned side) against the Gaussian
// product theorem, the la <= lb swap that lets one kernel serve both (s|p) and
// (p|s), one stream per class, and the device side matching the pinned side.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

namespace {

namespace fb = free_breakfast;

constexpr double kTol = 1e-15;

// Record k of a field, as written on the host.
template <typename T>
T at(const fb::MirroredBuffer<T> &field, const std::size_t k) {
  return field.pinned().data()[k];
}

// A field's device side, copied back to the host on the field's stream.
template <typename T>
std::vector<T> download(const fb::MirroredBuffer<T> &field) {
  std::vector<T> out(field.size());
  const wwr::wwrStream_t stream = field.handle()->stream();
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.device().data(), sizeof(T) * out.size(),
                                wwr::wwrMemcpyDeviceToHost, stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

// A field's pinned side.
template <typename T>
std::vector<T> pinned(const fb::MirroredBuffer<T> &field) {
  const T *p = field.pinned().data();
  return {p, p + field.size()};
}

std::vector<fb::PrimitivePairClass> classes_of(fb::DeviceResources &res,
                                               const fb::BasisSet &basis) {
  auto classes = fb::primitive_pair_classes(res, basis);
  EXPECT_TRUE(classes);
  return classes ? std::move(*classes) : std::vector<fb::PrimitivePairClass>{};
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
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, p_then_s());
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
  for (const fb::PrimitivePairClass &cls : classes) {
    const auto n = static_cast<std::size_t>(cls.n_rec);
    EXPECT_EQ(cls.zeta.size(), n);
    EXPECT_EQ(cls.pa_y.size(), n);
    EXPECT_EQ(cls.off_b.size(), n);
  }
}

TEST(Mmd, HigherMomentumBraIsSwappedToTheKet) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = p_then_s();
  const auto classes = classes_of(res, basis);
  ASSERT_EQ(classes.size(), 3U);
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
    EXPECT_EQ(at(sp.off_a, k), s_off) << "record " << k;
    EXPECT_EQ(at(sp.off_b, k), p_off) << "record " << k;
    EXPECT_NEAR(at(sp.zeta, k), zeta, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.eb, k), beta, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.p_z, k), pz, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.pa_z, k), pz - z_a, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.pb_z, k), pz - z_b, kTol) << "record " << k;
    // x and y: every center is on the z axis
    for (const fb::MirroredBuffer<double> *axis :
         {&sp.p_x, &sp.p_y, &sp.pa_x, &sp.pa_y, &sp.pb_x, &sp.pb_y}) {
      EXPECT_EQ(at(*axis, k), 0.0) << "record " << k;
    }
    const double s = c_a * ket[k].coefficient * std::pow(std::numbers::pi / zeta, 1.5);
    EXPECT_NEAR(at(sp.s, k), s, kTol * s) << "record " << k;
  }
}

TEST(Mmd, DiagonalShellPairCarriesEqualOffsets) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, p_then_s());
  ASSERT_EQ(classes.size(), 3U);
  for (const fb::PrimitivePairClass *cls : {&classes[0], &classes[2]}) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls->n_rec); ++k) {
      EXPECT_EQ(at(cls->off_a, k), at(cls->off_b, k)) << "(" << cls->la << ", " << cls->lb << ")";
    }
  }
  // Records are bra-primitive-major: (alpha_0, alpha_0), (alpha_0, alpha_1), ...
  const fb::PrimitivePairClass &pp = classes[2];
  EXPECT_NEAR(at(pp.zeta, 0), 4.0, kTol);
  EXPECT_NEAR(at(pp.zeta, 1), 2.5, kTol);
  EXPECT_NEAR(at(pp.zeta, 2), 2.5, kTol);
  EXPECT_NEAR(at(pp.zeta, 3), 1.0, kTol);
}

TEST(Mmd, EmptyBasisHasNoClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  EXPECT_TRUE(classes_of(res, fb::BasisSet{}).empty());
}

TEST(Mmd, EachClassHasItsOwnStream) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, p_then_s());
  std::set<wwr::wwrStream_t> streams{res.stream()};
  for (const fb::PrimitivePairClass &cls : classes) {
    EXPECT_EQ(cls.device_stream()->dev_idx(), res.dev_idx());
    EXPECT_TRUE(streams.insert(cls.device_stream()->stream()).second)
        << "class (" << cls.la << ", " << cls.lb << ") shares a stream";
  }
}

// What reaches the device is what was written on the host, bit for bit.
TEST(Mmd, DeviceSideMatchesThePinnedSide) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, p_then_s());
  for (const fb::PrimitivePairClass &cls : classes) {
    for (const fb::MirroredBuffer<double> *field :
         {&cls.zeta, &cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z, &cls.pb_x,
          &cls.pb_y, &cls.pb_z, &cls.s, &cls.eb}) {
      EXPECT_EQ(download(*field), pinned(*field)) << "(" << cls.la << ", " << cls.lb << ")";
    }
    for (const fb::MirroredBuffer<std::int32_t> *field : {&cls.off_a, &cls.off_b}) {
      EXPECT_EQ(download(*field), pinned(*field)) << "(" << cls.la << ", " << cls.lb << ")";
    }
  }
}

} // namespace
