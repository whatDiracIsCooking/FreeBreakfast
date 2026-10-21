// GPU cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field (on each class's pinned side) against the Gaussian
// product theorem, the la <= lb swap that lets one kernel serve both (s|p) and
// (p|s), one stream per class, the device side matching the pinned side, and
// each class's records in descending zeta order, equal zetas in the order the
// host wrote them: on shells whose exponents ascend, so the sort has to move
// records, and on a class past the merge sort's 2816-key tile.

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

// The classes, each stream synchronized: the order and the pinned side's
// refresh from the device are queued on it, so the pinned side holds the final
// records only after.
std::vector<fb::PrimitivePairClass> classes_of(fb::DeviceResources &res,
                                               const fb::BasisSet &basis) {
  auto classes = fb::primitive_pair_classes(res, basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  for (const fb::PrimitivePairClass &cls : *classes) {
    EXPECT_EQ(wwr::wwrStreamSynchronize(cls.device_stream()->stream()), wwr::wwrSuccess);
  }
  return std::move(*classes);
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

  // The p shell's exponents descend, so its records' zetas do too: record k is
  // ket primitive k.
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
  // Descending zeta, the tie (alpha_0, alpha_1) / (alpha_1, alpha_0) in the
  // order the host wrote it.
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

// One record of an s-only basis, as the host writes it.
struct SRecord {
  double zeta;
  double p_z;
  double pa_z;
  double pb_z;
  double s;
  double eb;
  std::int32_t off_a;
  std::int32_t off_b;
};

// An s shell on the z axis at z, its exponents as given, coefficients 1.
fb::Shell s_shell(const std::vector<double> &exponents, const double z) {
  return fb::Shell(0, exponents, std::vector<double>(exponents.size(), 1.0),
                   fb::Vec<double, 3>{0.0, 0.0, z});
}

// The (s|s) class's records in the order the device must give: every shell
// pair i <= j, alpha-major, as the host writes them, then stably sorted by
// zeta, descending.
std::vector<SRecord> want_s_records(const fb::BasisSet &basis) {
  std::vector<SRecord> out;
  const std::vector<fb::Shell> &shells = basis.shells();
  for (std::size_t i = 0; i < shells.size(); ++i) {
    for (std::size_t j = i; j < shells.size(); ++j) {
      const double za = shells[i].center().z;
      const double zb = shells[j].center().z;
      for (const fb::Primitive &pa : shells[i].primitives()) {
        for (const fb::Primitive &pb : shells[j].primitives()) {
          const double zeta = pa.exponent + pb.exponent;
          const double pz = ((pa.exponent * za) + (pb.exponent * zb)) / zeta;
          out.push_back({zeta, pz, pz - za, pz - zb,
                         pa.coefficient * pb.coefficient * std::pow(std::numbers::pi / zeta, 1.5),
                         pb.exponent, static_cast<std::int32_t>(basis.ao_offset_cart(i)),
                         static_cast<std::int32_t>(basis.ao_offset_cart(j))});
        }
      }
    }
  }
  std::ranges::stable_sort(out, std::greater<>{}, &SRecord::zeta);
  return out;
}

void expect_s_records(const fb::BasisSet &basis) {
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, basis);
  ASSERT_EQ(classes.size(), 1U);
  const fb::PrimitivePairClass &ss = classes[0];
  const std::vector<SRecord> want = want_s_records(basis);
  ASSERT_EQ(static_cast<std::size_t>(ss.n_rec), want.size());
  for (std::size_t k = 0; k < want.size(); ++k) {
    // zeta is a + b in either order, so exactly the host's; the order of
    // equal zetas is pinned by the offsets and eb.
    ASSERT_EQ(at(ss.zeta, k), want[k].zeta) << "record " << k;
    ASSERT_EQ(at(ss.off_a, k), want[k].off_a) << "record " << k;
    ASSERT_EQ(at(ss.off_b, k), want[k].off_b) << "record " << k;
    ASSERT_EQ(at(ss.eb, k), want[k].eb) << "record " << k;
    ASSERT_NEAR(at(ss.p_z, k), want[k].p_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.pa_z, k), want[k].pa_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.pb_z, k), want[k].pb_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.s, k), want[k].s, kTol * want[k].s) << "record " << k;
  }
  // The device side was permuted, and the pinned side refreshed from it.
  EXPECT_EQ(download(ss.zeta), pinned(ss.zeta));
  EXPECT_EQ(download(ss.off_b), pinned(ss.off_b));
}

TEST(Mmd, RecordsAreInDescendingZetaOrder) {
  FBF_REQUIRE_GPU();
  // Ascending exponents, so the order the host writes is ascending in zeta
  // within each shell pair, and the sort must move nearly every record.
  expect_s_records(fb::BasisSet({s_shell({0.3, 1.1, 4.0}, 0.0), s_shell({0.7, 2.5}, 1.2)}));
}

TEST(Mmd, RecordsAreInDescendingZetaOrderPastOneSortTile) {
  FBF_REQUIRE_GPU();
  // One s shell of 60 ascending exponents against itself: 3600 records, past
  // the merge sort's 2816-key tile, so its merge pass runs.
  std::vector<double> exponents;
  for (std::int32_t i = 0; i < 60; ++i) {
    exponents.push_back(0.1 * std::pow(1.2, i));
  }
  expect_s_records(fb::BasisSet({s_shell(exponents, 0.4)}));
}

} // namespace
