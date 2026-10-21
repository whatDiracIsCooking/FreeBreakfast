// GPU cases for integrals.coulomb: the Cartesian Coulomb matrix J of H2O in
// STO-3G (s, p; O's p shell precedes the H s shells, so the la <= lb swap is
// exercised, in the density's read as in J's scatter) and cc-pVDZ (s, p, d, so
// every e_hermite class through (4, 4)) against libcint's J of the same
// density, recorded by PySCF (test/gtest/data/coulomb/, copied from
// FelizJueves's fixtures by tools/extract_integral_golden.py), both unscreened
// (tau = 0) and at the default screen; J from coulomb_pair_classes' pruned
// records against the same goldens (the H10 chain's too, unscreened), their
// count pinned, and the pruned records a subsequence of mmd's; that a tau no
// bound reaches screens everything; plus, as overlap_gpu.cpp, the zeroing and
// the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.linalg.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// FelizJueves holds its J to libcint at 1e-12 of scale (test/oracle/
// tolerance.py); the same bar here, against max|J| as overlap_gpu.cpp. The
// recorded density is a Hilbert matrix, every element positive, so max|J| is
// not a scale cancellation has shrunk.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> density; // nao x nao, row-major; each is symmetric
  std::vector<double> vj;
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "coulomb" / (stem + ".json"));
  EXPECT_TRUE(doc) << stem;
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
  const auto &arrays = (*doc)["arrays"];
  Golden g{fb::BasisSet(std::move(shells)), arrays["density"]["values"].get<std::vector<double>>(),
           arrays["vj"]["values"].get<std::vector<double>>()};
  EXPECT_EQ(g.basis.nao_cart(), (*doc)["nao"].get<std::uint32_t>()) << stem;
  return g;
}

std::vector<double> download(fb::DeviceResources &res, const fb::CartesianMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// A symmetric matrix onto the device: row- and column-major are one layout.
void upload(fb::DeviceResources &res, const std::vector<double> &host, fb::CartesianMatrixAO &m) {
  ASSERT_EQ(host.size(), m.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

// J of g's density at tau, downloaded.
std::vector<double> golden_j(const Golden &g, const double tau) {
  fb::DeviceResources res{0};
  const auto classes = fb::primitive_pair_classes(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  fb::CartesianMatrixAO j(res, g.basis);
  const fb::Status status = fb::coulomb(res, *classes, density, j, tau);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, j);
}

class CoulombGolden : public ::testing::TestWithParam<const char *> {};

// At tau = 0 nothing is screened; at the default each skipped term of a J_p is
// under 1e-10, and the screened J must still meet the same bar against
// libcint. How far the screen moved J is recorded, not bounded: it is not
// bounded by tau (jab weights each J_p term by the bra's E^ab_p).
TEST_P(CoulombGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vj.empty());
  double scale = 0.0;
  for (const double v : g.vj) {
    scale = std::max(scale, std::abs(v));
  }
  const std::size_t n = g.basis.nao_cart();
  const std::vector<double> unscreened = golden_j(g, 0.0);
  const std::vector<double> screened = golden_j(g, 1e-10);
  for (const auto &[name, got] : {std::pair{"unscreened", &unscreened}, {"screened", &screened}}) {
    ASSERT_EQ(got->size(), g.vj.size()) << name;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < n; ++k) {
        const double err = std::abs((*got)[i * n + k] - g.vj[i * n + k]);
        worst = std::max(worst, err);
        EXPECT_LE(err, kRelTol * scale) << name << " J(" << i << ", " << k << ")";
        // The atomics add each mirror pair in their own order, so symmetric to
        // rounding, not bitwise.
        EXPECT_LE(std::abs((*got)[i * n + k] - (*got)[k * n + i]), kRelTol * scale) << name;
      }
    }
    RecordProperty(std::format("{}_worst_abs_error", name), std::format("{:.3e}", worst));
  }
  double moved = 0.0;
  for (std::size_t i = 0; i < n * n; ++i) {
    moved = std::max(moved, std::abs(screened[i] - unscreened[i]));
  }
  RecordProperty("screen_max_abs_change", std::format("{:.3e}", moved));
}

INSTANTIATE_TEST_SUITE_P(H2O, CoulombGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

std::size_t record_count(const std::vector<fb::PrimitivePairClass> &classes) {
  std::size_t n = 0;
  for (const fb::PrimitivePairClass &cls : classes) {
    n += static_cast<std::size_t>(cls.n_rec);
  }
  return n;
}

// J of g's density at tau from coulomb_pair_classes' records pruned at prune
// (its default unless given), downloaded, and the record counts before and
// after.
struct PrunedJ {
  std::vector<double> j;
  std::size_t before;
  std::size_t after;
};

PrunedJ pruned_j(const Golden &g, const double tau, const double prune = 1e-13) {
  fb::DeviceResources res{0};
  const auto classes = fb::primitive_pair_classes(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  const auto pruned = fb::coulomb_pair_classes(res, *classes, prune);
  EXPECT_TRUE(pruned) << (pruned ? "" : pruned.error().message);
  if (!pruned) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  fb::CartesianMatrixAO j(res, g.basis);
  const fb::Status status = fb::coulomb(res, *pruned, density, j, tau);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return {download(res, j), record_count(*classes), record_count(*pruned)};
}

struct PruneCase {
  const char *stem;
  // The records coulomb_pair_classes keeps at its default, of mmd's.
  std::size_t kept;
  std::size_t records;
};

// The molecule, not the struct's bytes, in the test's name and in ctest's.
void PrintTo(const PruneCase &c, std::ostream *os) {
  *os << c.stem;
}

class CoulombPruned : public ::testing::TestWithParam<PruneCase> {};

// The pruned records give libcint's J to the same bar as every record: at
// tau = 0 on all three molecules, and at the default screen on H2O (the H10
// chain's screened J misses the bar with every record too, #132's "Not in
// scope"). The record counts are pinned; how far pruning moved J (against
// every record, both at tau = 0) is recorded.
TEST_P(CoulombPruned, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const PruneCase c = GetParam();
  const Golden g = read_golden(c.stem);
  ASSERT_FALSE(g.vj.empty());
  double scale = 0.0;
  for (const double v : g.vj) {
    scale = std::max(scale, std::abs(v));
  }
  const bool chain = std::string_view{c.stem} == "h10_chain_ccpvdz";
  const PrunedJ unscreened = pruned_j(g, 0.0);
  EXPECT_EQ(unscreened.before, c.records);
  EXPECT_EQ(unscreened.after, c.kept);
  std::vector<std::pair<const char *, const std::vector<double> *>> runs{
      {"unscreened", &unscreened.j}};
  PrunedJ screened;
  if (!chain) {
    screened = pruned_j(g, 1e-10);
    runs.emplace_back("screened", &screened.j);
  }
  const std::size_t n = g.basis.nao_cart();
  for (const auto &[name, got] : runs) {
    ASSERT_EQ(got->size(), g.vj.size()) << name;
    double worst = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      const double err = std::abs((*got)[i] - g.vj[i]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale) << name << " J(" << i / n << ", " << i % n << ")";
    }
    RecordProperty(std::format("{}_worst_abs_error", name), std::format("{:.3e}", worst));
  }
  const std::vector<double> every = golden_j(g, 0.0);
  double moved = 0.0;
  for (std::size_t i = 0; i < n * n; ++i) {
    moved = std::max(moved, std::abs(unscreened.j[i] - every[i]));
  }
  RecordProperty("prune_max_abs_change", std::format("{:.3e}", moved));
}

INSTANTIATE_TEST_SUITE_P(Molecules, CoulombPruned,
                         ::testing::Values(PruneCase{"h2o_sto3g", 135, 135},
                                           PruneCase{"h2o_ccpvdz", 572, 593},
                                           PruneCase{"h10_chain_ccpvdz", 842, 1305}),
                         [](const auto &info) { return std::string{info.param.stem}; });

// Every kept record is one of mmd's, every field carried, in mmd's order; at
// tau = 0 every record is kept, and a tau above every bound keeps none (and
// leaves out every emptied class).
TEST(CoulombPairClasses, KeepsASubsequenceOfMmdsRecords) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h10_chain_ccpvdz");
  ASSERT_FALSE(g.vj.empty());
  fb::DeviceResources res{0};
  auto classes = fb::primitive_pair_classes(res, g.basis);
  ASSERT_TRUE(classes);
  for (fb::PrimitivePairClass &cls : *classes) {
    ASSERT_EQ(wwr::wwrStreamSynchronize(cls.device_stream()->stream()), wwr::wwrSuccess);
  }
  auto pruned = fb::coulomb_pair_classes(res, *classes);
  ASSERT_TRUE(pruned);
  ASSERT_LE(pruned->size(), classes->size());
  std::size_t from = 0;
  for (fb::PrimitivePairClass &out : *pruned) {
    // The device side, not the pinned one the copy was written from.
    std::vector<double> zeta(static_cast<std::size_t>(out.n_rec));
    std::vector<double> p_z(zeta.size());
    std::vector<double> pa_z(zeta.size());
    std::vector<double> pb_z(zeta.size());
    std::vector<double> s(zeta.size());
    std::vector<double> eb(zeta.size());
    std::vector<double> bound(zeta.size());
    std::vector<std::int32_t> off_a(zeta.size());
    std::vector<std::int32_t> off_b(zeta.size());
    const auto get = [&](auto &host, const auto &field) {
      ASSERT_EQ(wwr::wwrMemcpy(host.data(), field.device().data(), sizeof(host[0]) * host.size(),
                               wwr::wwrMemcpyDeviceToHost),
                wwr::wwrSuccess);
    };
    ASSERT_EQ(wwr::wwrStreamSynchronize(out.device_stream()->stream()), wwr::wwrSuccess);
    get(zeta, out.zeta);
    get(p_z, out.p_z);
    get(pa_z, out.pa_z);
    get(pb_z, out.pb_z);
    get(s, out.s);
    get(eb, out.eb);
    get(bound, out.bound);
    get(off_a, out.off_a);
    get(off_b, out.off_b);
    // The class it came from: the first of the same momenta at or after from.
    while (from < classes->size() &&
           ((*classes)[from].la != out.la || (*classes)[from].lb != out.lb)) {
      ++from;
    }
    ASSERT_LT(from, classes->size()) << out.la << out.lb;
    fb::PrimitivePairClass &src = (*classes)[from++];
    std::int32_t k = 0;
    for (std::size_t r = 0; r < zeta.size(); ++r) {
      // Advance through src to the record r was copied from.
      const auto same = [&](const std::int32_t q) {
        const auto at = [q](auto &field) { return field.pinned().data()[q]; };
        return at(src.zeta) == zeta[r] && at(src.p_z) == p_z[r] && at(src.pa_z) == pa_z[r] &&
               at(src.pb_z) == pb_z[r] && at(src.s) == s[r] && at(src.eb) == eb[r] &&
               at(src.bound) == bound[r] && at(src.off_a) == off_a[r] && at(src.off_b) == off_b[r];
      };
      while (k < src.n_rec && !same(k)) {
        ++k;
      }
      ASSERT_LT(k, src.n_rec) << "class (" << out.la << ", " << out.lb << ") record " << r
                              << " is not one of mmd's, in order";
      ++k;
    }
  }

  const auto all = fb::coulomb_pair_classes(res, *classes, 0.0);
  ASSERT_TRUE(all);
  EXPECT_EQ(all->size(), classes->size());
  EXPECT_EQ(record_count(*all), record_count(*classes));
  const auto none = fb::coulomb_pair_classes(res, *classes, 1e300);
  ASSERT_TRUE(none);
  EXPECT_TRUE(none->empty());
}

TEST(CoulombPairClasses, RefusesANegativeOrNaNTauAndAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet s_shell({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = fb::primitive_pair_classes(res, s_shell);
  ASSERT_TRUE(classes);
  for (const double tau : {-1e-13, std::numeric_limits<double>::quiet_NaN()}) {
    const auto out = fb::coulomb_pair_classes(res, *classes, tau);
    ASSERT_FALSE(out) << tau;
    EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig) << tau;
  }
  const fb::BasisSet f_shell({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto f_classes = fb::primitive_pair_classes(res, f_shell);
  ASSERT_TRUE(f_classes);
  const auto out = fb::coulomb_pair_classes(res, *f_classes);
  ASSERT_FALSE(out);
  EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(Coulomb, ZeroesTheMatrixFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(j.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  // No classes: the only work left is the zeroing.
  ASSERT_TRUE(fb::coulomb(res, {}, density, j));
  EXPECT_EQ(download(res, j)[0], 0.0);
}

// A tau above every Dmax * Ip * Iq breaks every lane at its first ket, so J is
// exactly zero: the screen reaches the kernel and fires.
TEST(Coulomb, ATauAboveEveryBoundScreensEverything) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h2o_sto3g");
  ASSERT_FALSE(g.vj.empty());
  for (const double v : golden_j(g, 1e300)) {
    ASSERT_EQ(v, 0.0);
  }
}

TEST(Coulomb, RefusesANegativeOrNaNTau) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  for (const double tau : {-1e-10, std::numeric_limits<double>::quiet_NaN()}) {
    const fb::Status status = fb::coulomb(res, *classes, density, j, tau);
    ASSERT_FALSE(status) << tau;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig) << tau;
  }
}

TEST(Coulomb, RefusesADensityOfAnotherOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                          fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  const auto classes = fb::primitive_pair_classes(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO j(res, two);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(Coulomb, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
