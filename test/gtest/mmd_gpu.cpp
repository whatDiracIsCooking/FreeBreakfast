// GPU cases for integrals.mmd: the primitive-pair records a basis turns into,
// checked field by field (each class's device side, copied back) against the
// Gaussian product theorem, the la <= lb swap that lets one kernel serve both
// (s|p) and (p|s), one stream per class, a diagonal shell pair's mirrored
// records bitwise mirrored, and each class's records in descending zeta order, equal zetas in
// the shell-pair, then bra-primitive, order: on shells whose exponents ascend, so the sort has
// to move
// records, and on a class past the merge sort's 2816-key tile; and each
// record's Schwarz bound, formed on the device, against a host evaluation of
// its closed form for every class of H2O/cc-pVDZ; and what the two-electron
// builds derive from: the exponents and shell ranks against the basis, and the
// dressed bound as stored and swapped against the one ordered_pair_classes
// forms for the same record.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.integral_direct.exchange.pairs;

#include "mmd_records.h"

namespace {

namespace fb = free_breakfast;
using Records = fb::test::MmdRecords;

constexpr double kTol = 1e-15;

// Record k of a field, copied back.
template <typename T>
T at(const std::vector<T> &field, const std::size_t k) {
  return field[k];
}

// The classes, each one copied back: the records exist on the device only.
std::vector<Records> classes_of(fb::DeviceResources &res, const fb::BasisSet &basis) {
  auto classes = fb::primitive_pair_classes(res, basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  return fb::test::copy_back(*classes);
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
  for (const Records &cls : classes) {
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
  const Records &sp = classes[1];
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
    EXPECT_EQ(at(sp.ea, k), alpha) << "record " << k;
    EXPECT_NEAR(at(sp.eb, k), beta, kTol) << "record " << k;
    // Each the first, and only, shell of its momentum.
    EXPECT_EQ(at(sp.rank_a, k), 0) << "record " << k;
    EXPECT_EQ(at(sp.rank_b, k), 0) << "record " << k;
    EXPECT_NEAR(at(sp.p_z, k), pz, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.pa_z, k), pz - z_a, kTol) << "record " << k;
    EXPECT_NEAR(at(sp.pb_z, k), pz - z_b, kTol) << "record " << k;
    // x and y: every center is on the z axis
    for (const std::vector<double> *axis :
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
  for (const Records *cls : {&classes[0], &classes[2]}) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls->n_rec); ++k) {
      EXPECT_EQ(at(cls->off_a, k), at(cls->off_b, k)) << "(" << cls->la << ", " << cls->lb << ")";
    }
  }
  // Descending zeta, the tie (alpha_0, alpha_1) / (alpha_1, alpha_0) in the
  // order the host wrote it.
  const Records &pp = classes[2];
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
  const auto classes = fb::primitive_pair_classes(res, p_then_s());
  ASSERT_TRUE(classes);
  std::set<wwr::wwrStream_t> streams{res.stream()};
  for (const fb::PrimitivePairClass &cls : *classes) {
    EXPECT_EQ(cls.device_stream()->dev_idx(), res.dev_idx());
    EXPECT_TRUE(streams.insert(cls.device_stream()->stream()).second)
        << "class (" << cls.la << ", " << cls.lb << ") shares a stream";
  }
}

// One record of an s-only basis, as the host writes it.
struct SRecord {
  double zeta;
  double p_z;
  double pa_z;
  double pb_z;
  double s;
  double ea;
  double eb;
  std::int32_t off_a;
  std::int32_t off_b;
  // An s-only basis: each shell's rank is its index.
  std::int32_t rank_a;
  std::int32_t rank_b;
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
                         pa.exponent, pb.exponent,
                         static_cast<std::int32_t>(basis.ao_offset_cart(i)),
                         static_cast<std::int32_t>(basis.ao_offset_cart(j)),
                         static_cast<std::int32_t>(i), static_cast<std::int32_t>(j)});
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
  const Records &ss = classes[0];
  const std::vector<SRecord> want = want_s_records(basis);
  ASSERT_EQ(static_cast<std::size_t>(ss.n_rec), want.size());
  for (std::size_t k = 0; k < want.size(); ++k) {
    // zeta is a + b in either order, so exactly the host's; the order of
    // equal zetas is pinned by the offsets and eb.
    ASSERT_EQ(at(ss.zeta, k), want[k].zeta) << "record " << k;
    ASSERT_EQ(at(ss.off_a, k), want[k].off_a) << "record " << k;
    ASSERT_EQ(at(ss.off_b, k), want[k].off_b) << "record " << k;
    ASSERT_EQ(at(ss.ea, k), want[k].ea) << "record " << k;
    ASSERT_EQ(at(ss.eb, k), want[k].eb) << "record " << k;
    ASSERT_EQ(at(ss.rank_a, k), want[k].rank_a) << "record " << k;
    ASSERT_EQ(at(ss.rank_b, k), want[k].rank_b) << "record " << k;
    ASSERT_NEAR(at(ss.p_z, k), want[k].p_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.pa_z, k), want[k].pa_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.pb_z, k), want[k].pb_z, kTol) << "record " << k;
    ASSERT_NEAR(at(ss.s, k), want[k].s, kTol * want[k].s) << "record " << k;
  }
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

// A diagonal shell pair's record (alpha_i, alpha_j) and its mirror
// (alpha_j, alpha_i) carry the same P bit for bit, so each one's P - A is the
// other's P - B: the device forms P with no FMA contraction, so its two
// products are rounded on their own and their sum commutes exactly. K's gather
// and its test lean on it (a diagonal record's swap is a record of its own
// class). Off the axes, on a d shell, so every component and a centre whose
// products would round differently under a fused multiply-add are in play.
TEST(Mmd, DiagonalMirrorsAreBitwiseMirrored) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Vec<double, 3> c{0.37, -1.29, 2.71};
  const auto classes =
      classes_of(res, fb::BasisSet({fb::Shell(2, {{3.3, 0.2}, {1.17, 0.5}, {0.29, 0.7}}, c)}));
  ASSERT_EQ(classes.size(), 1U);
  const Records &dd = classes[0];
  ASSERT_EQ(dd.n_rec, 9);
  std::map<std::pair<double, double>, std::size_t> by_exponents;
  for (std::size_t k = 0; k < 9; ++k) {
    by_exponents[{dd.ea[k], dd.eb[k]}] = k;
  }
  ASSERT_EQ(by_exponents.size(), 9U);
  for (std::size_t k = 0; k < 9; ++k) {
    const std::size_t m = by_exponents.at({dd.eb[k], dd.ea[k]});
    EXPECT_EQ(dd.zeta[k], dd.zeta[m]) << "record " << k;
    EXPECT_EQ(dd.s[k], dd.s[m]) << "record " << k;
    const std::array<double, 9> mine{dd.p_x[k],  dd.p_y[k],  dd.p_z[k],  dd.pa_x[k], dd.pa_y[k],
                                     dd.pa_z[k], dd.pb_x[k], dd.pb_y[k], dd.pb_z[k]};
    const std::array<double, 9> mirror{dd.p_x[m],  dd.p_y[m],  dd.p_z[m],  dd.pb_x[m], dd.pb_y[m],
                                       dd.pb_z[m], dd.pa_x[m], dd.pa_y[m], dd.pa_z[m]};
    EXPECT_EQ(mine, mirror) << "record " << k;
  }
}

// k!!, with (-1)!! = 1.
double double_factorial(const std::int32_t k) {
  double out = 1.0;
  for (std::int32_t i = k; i > 1; i -= 2) {
    out *= i;
  }
  return out;
}

// The record bound I = pair_bound(zeta, l) in its closed form at PQ = 0
// (FelizJueves's kernel_pair_bound.py): with x = sqrt(zeta),
// I = sqrt(x) SUM_{n <= l} D_n x^n, D_n = (2 / sqrt(pi))^{1/2} 2^{-1/4}
// SUM_{|p| = n} sqrt(PROD_d (2 p_d - 1)!! / (2n + 1)), p every Hermite index.
// Summed term by term here, not by the kernel's Horner fold of folded D_n.
double host_pair_bound(const double zeta, const std::int32_t l) {
  const double x = std::sqrt(zeta);
  const double k = std::sqrt(2.0 / std::sqrt(std::numbers::pi)) * std::pow(2.0, -0.25);
  double sum = 0.0;
  for (std::int32_t px = 0; px <= l; ++px) {
    for (std::int32_t py = 0; px + py <= l; ++py) {
      for (std::int32_t pz = 0; px + py + pz <= l; ++pz) {
        const std::int32_t n = px + py + pz;
        const double d = std::sqrt(double_factorial((2 * px) - 1) * double_factorial((2 * py) - 1) *
                                   double_factorial((2 * pz) - 1) / ((2.0 * n) + 1.0));
        sum += d * std::pow(x, n);
      }
    }
  }
  return k * std::sqrt(x) * sum;
}

// H2O/cc-pVDZ (s, p, d: all six classes), as coulomb_gpu.cpp reads it.
fb::BasisSet h2o_ccpvdz() {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "coulomb" / "h2o_ccpvdz.json");
  EXPECT_TRUE(doc);
  if (!doc) {
    return {};
  }
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(),
                        fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  return fb::BasisSet(std::move(shells));
}

// Each record's bound, copied back from the device,
// within 16 eps of the host's closed form, and non-increasing within each
// class: the bound rises with zeta, and the records are in descending zeta.
TEST(Mmd, BoundMatchesTheClosedFormAndFollowsTheZetaOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto classes = classes_of(res, h2o_ccpvdz());
  ASSERT_EQ(classes.size(), 6U);
  constexpr double kBoundTol = 16.0 * std::numeric_limits<double>::epsilon();
  for (const Records &cls : classes) {
    const std::int32_t l = cls.la + cls.lb;
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
      const double want = host_pair_bound(at(cls.zeta, k), l);
      ASSERT_NEAR(at(cls.bound, k), want, kBoundTol * want)
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      if (k > 0) {
        ASSERT_LE(at(cls.bound, k), at(cls.bound, k - 1))
            << "(" << cls.la << ", " << cls.lb << ") record " << k;
      }
    }
  }
}

// Each record's exponents and shell ranks match the basis: its two shells
// named by their AO offsets, of the class's momenta, ea and eb exponents of
// theirs summing to zeta exactly (the host formed zeta as ea + eb), and each
// rank the shell's index among the basis's shells of its momentum.
TEST(Mmd, ExponentsAndRanksMatchTheBasis) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o_ccpvdz();
  const auto classes = classes_of(res, basis);
  ASSERT_EQ(classes.size(), 6U);
  const std::vector<fb::Shell> &shells = basis.shells();
  std::map<std::int32_t, std::size_t> shell_at;
  std::vector<std::int32_t> rank(shells.size());
  std::map<std::int32_t, std::int32_t> next_rank;
  for (std::size_t i = 0; i < shells.size(); ++i) {
    shell_at[static_cast<std::int32_t>(basis.ao_offset_cart(i))] = i;
    rank[i] = next_rank[shells[i].angular_momentum()]++;
  }
  const auto has_exponent = [](const fb::Shell &sh, const double e) {
    return std::ranges::any_of(sh.primitives(),
                               [e](const fb::Primitive &p) { return p.exponent == e; });
  };
  for (const Records &cls : classes) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
      ASSERT_TRUE(shell_at.contains(at(cls.off_a, k)) && shell_at.contains(at(cls.off_b, k)));
      const std::size_t a = shell_at.at(at(cls.off_a, k));
      const std::size_t b = shell_at.at(at(cls.off_b, k));
      ASSERT_EQ(shells[a].angular_momentum(), cls.la);
      ASSERT_EQ(shells[b].angular_momentum(), cls.lb);
      ASSERT_TRUE(has_exponent(shells[a], at(cls.ea, k)))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_TRUE(has_exponent(shells[b], at(cls.eb, k)))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_EQ(at(cls.ea, k) + at(cls.eb, k), at(cls.zeta, k))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_EQ(at(cls.rank_a, k), rank[a]) << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_EQ(at(cls.rank_b, k), rank[b]) << "(" << cls.la << ", " << cls.lb << ") record " << k;
    }
  }
}

// What an ordered record is known by: its first and second shells' AO
// offsets, zeta, P - A (A the first shell's centre), P - B and s. A record and
// its copy in an ordered class carry these bit for bit.
using OrderedKey = std::tuple<std::int32_t, std::int32_t, double, double, double, double, double,
                              double, double, double>;

// bound_ab and bound_ba bit for bit against the dressed bound
// ordered_pair_classes forms for the same record as stored and swapped: the
// same generated kernel on the same fields. A diagonal record has no swapped
// copy there, but its swap is, field for field, a record of its own class (the
// primitives the other way round), so it is found all the same.
TEST(Mmd, DressedBoundsMatchTheOrderedClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = h2o_ccpvdz();
  const auto mmd = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(mmd);
  ASSERT_EQ(mmd->size(), 6U);
  const auto ordered = fb::ordered_pair_classes(res, basis, *mmd);
  ASSERT_TRUE(ordered);
  const std::vector<Records> classes = fb::test::copy_back(*mmd);

  std::map<OrderedKey, double> bound_of;
  for (const fb::OrderedPairClass &cls : *ordered) {
    // The ordered class is device only: each field copied back on its stream.
    const auto back = [&cls]<typename T>(const fb::StreamBuffers::device<T> &field) {
      std::vector<T> out(static_cast<std::size_t>(cls.n_rec));
      const wwr::wwrStream_t stream = cls.device_stream()->stream();
      EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.data(), sizeof(T) * out.size(),
                                    wwr::wwrMemcpyDeviceToHost, stream),
                wwr::wwrSuccess);
      EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
      return out;
    };
    const auto off_a = back(cls.off_a);
    const auto off_b = back(cls.off_b);
    const auto zeta = back(cls.zeta);
    const auto pa_x = back(cls.pa_x);
    const auto pa_y = back(cls.pa_y);
    const auto pa_z = back(cls.pa_z);
    const auto pb_x = back(cls.pb_x);
    const auto pb_y = back(cls.pb_y);
    const auto pb_z = back(cls.pb_z);
    const auto s = back(cls.s);
    const auto record_bound = back(cls.record_bound);
    for (std::size_t r = 0; r < static_cast<std::size_t>(cls.n_rec); ++r) {
      const OrderedKey key{off_a[r], off_b[r], zeta[r], pa_x[r], pa_y[r],
                           pa_z[r],  pb_x[r],  pb_y[r], pb_z[r], s[r]};
      const auto [it, fresh] = bound_of.emplace(key, record_bound[r]);
      // Two records alike in every field the bound reads have one bound.
      ASSERT_TRUE(fresh || it->second == record_bound[r]);
    }
  }

  std::size_t positive = 0;
  for (const Records &cls : classes) {
    for (std::size_t k = 0; k < static_cast<std::size_t>(cls.n_rec); ++k) {
      const OrderedKey as_stored{
          at(cls.off_a, k), at(cls.off_b, k), at(cls.zeta, k), at(cls.pa_x, k), at(cls.pa_y, k),
          at(cls.pa_z, k),  at(cls.pb_x, k),  at(cls.pb_y, k), at(cls.pb_z, k), at(cls.s, k)};
      const OrderedKey swapped{at(cls.off_b, k), at(cls.off_a, k), at(cls.zeta, k), at(cls.pb_x, k),
                               at(cls.pb_y, k),  at(cls.pb_z, k),  at(cls.pa_x, k), at(cls.pa_y, k),
                               at(cls.pa_z, k),  at(cls.s, k)};
      ASSERT_TRUE(bound_of.contains(as_stored))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_TRUE(bound_of.contains(swapped))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_EQ(at(cls.bound_ab, k), bound_of.at(as_stored))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      ASSERT_EQ(at(cls.bound_ba, k), bound_of.at(swapped))
          << "(" << cls.la << ", " << cls.lb << ") record " << k;
      positive += at(cls.bound_ab, k) > 0.0 ? 1 : 0;
    }
  }
  // Not vacuous: the bounds were formed, not left at their zeroed start.
  EXPECT_GT(positive, 0U);
}

} // namespace
