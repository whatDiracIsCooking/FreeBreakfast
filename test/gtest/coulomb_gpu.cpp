// GPU cases for integrals.eri.integral_direct.coulomb: the Cartesian Coulomb
// matrix J of H2O in STO-3G (s, p; O's p shell precedes the H s shells, so the
// la <= lb swap is exercised, in the density's read as in J's scatter) and
// cc-pVDZ (s, p, d, so every e_hermite class through (4, 4)) against libcint's
// J of the same density, recorded by PySCF (test/gtest/data/coulomb/, copied
// from FelizJueves's fixtures by tools/extract_integral_golden.py), both
// unscreened (tau = 0) and at the default screen, from every record (J's
// classes pruned at 0); J from coulomb_pair_classes' pruned records against the
// same goldens (the H10 chain's too, unscreened), their count pinned, and the
// device prune's records exactly the host prune's; that a tau no bound reaches
// screens everything; plus, as overlap_gpu.cpp, the zeroing and the refusals.

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

// J of g's density at tau from every record (J's classes pruned at 0),
// downloaded.
std::vector<double> golden_j(const Golden &g, const double tau) {
  fb::DeviceResources res{0};
  const auto classes = fb::primitive_pair_classes(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  auto every = fb::coulomb_pair_classes(res, *classes, 0.0);
  EXPECT_TRUE(every);
  if (!every) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  fb::CartesianMatrixAO j(res, g.basis);
  const fb::Status status = fb::coulomb(res, *every, density, j, tau);
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

// mmd's classes or J's.
template <typename Class>
std::size_t record_count(const std::vector<Class> &classes) {
  std::size_t n = 0;
  for (const Class &cls : classes) {
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
  auto pruned = fb::coulomb_pair_classes(res, *classes, prune);
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

// The host prune coulomb_pair_classes made before it moved to the device: B
// mmd's bound_ab, K = w ncart(la) B, the maxima over every record, and a
// record kept when `B Dc maxK >= tau || K Dc maxB >= tau`, Dc = 2. Each
// class's kept record indices, from mmd's pinned side (each field the test
// reads copied back from the device, the classes' streams synchronized).
std::vector<std::vector<std::int32_t>>
host_prune(const std::vector<fb::PrimitivePairClass> &classes, const double tau) {
  constexpr double kCap = 2.0;
  double max_bra = 0.0;
  double max_ket = 0.0;
  const auto ket = [](const fb::PrimitivePairClass &cls, const std::int32_t r) {
    const double w = cls.off_a.pinned().data()[r] == cls.off_b.pinned().data()[r] ? 1.0 : 2.0;
    return w * static_cast<double>(fb::function_count_cart(cls.la)) *
           cls.bound_ab.pinned().data()[r];
  };
  for (const fb::PrimitivePairClass &cls : classes) {
    for (std::int32_t r = 0; r < cls.n_rec; ++r) {
      max_bra = std::max(max_bra, cls.bound_ab.pinned().data()[r]);
      max_ket = std::max(max_ket, ket(cls, r));
    }
  }
  std::vector<std::vector<std::int32_t>> keep(classes.size());
  for (std::size_t c = 0; c < classes.size(); ++c) {
    const fb::PrimitivePairClass &cls = classes[c];
    for (std::int32_t r = 0; r < cls.n_rec; ++r) {
      if (cls.bound_ab.pinned().data()[r] * kCap * max_ket >= tau ||
          ket(cls, r) * kCap * max_bra >= tau) {
        keep[c].push_back(r);
      }
    }
  }
  return keep;
}

class CoulombPairClassesPrune : public ::testing::TestWithParam<const char *> {};

// The device prune keeps exactly the host prune's records: each J class is
// the host's kept records of the mmd class it came from, every field J reads
// bitwise, in mmd's order, an emptied class left out; at tau = 0 every
// record, and a tau above every bound none.
TEST_P(CoulombPairClassesPrune, KeepsTheHostPrunesRecords) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vj.empty());
  fb::DeviceResources res{0};
  auto classes = fb::primitive_pair_classes(res, g.basis);
  ASSERT_TRUE(classes);
  // mmd's pinned side is staging only: copy back what the host prune and the
  // checks below read.
  for (fb::PrimitivePairClass &cls : *classes) {
    for (fb::MirroredBuffer<double> *field :
         {&cls.zeta, &cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z, &cls.pb_x,
          &cls.pb_y, &cls.pb_z, &cls.s, &cls.bound, &cls.bound_ab}) {
      ASSERT_EQ(field->to_pinned(), wwr::wwrSuccess);
    }
    for (fb::MirroredBuffer<std::int32_t> *field : {&cls.off_a, &cls.off_b}) {
      ASSERT_EQ(field->to_pinned(), wwr::wwrSuccess);
    }
    ASSERT_EQ(wwr::wwrStreamSynchronize(cls.device_stream()->stream()), wwr::wwrSuccess);
  }
  for (const double tau : {1e-13, 1e-10, 0.0}) {
    const auto pruned = fb::coulomb_pair_classes(res, *classes, tau);
    ASSERT_TRUE(pruned) << tau;
    const std::vector<std::vector<std::int32_t>> keep = host_prune(*classes, tau);
    std::size_t k = 0;
    for (std::size_t c = 0; c < classes->size(); ++c) {
      if (keep[c].empty()) {
        continue;
      }
      ASSERT_LT(k, pruned->size()) << tau;
      const fb::PrimitivePairClass &src = (*classes)[c];
      const fb::CoulombPairClass &out = (*pruned)[k++];
      ASSERT_EQ(out.la, src.la) << tau;
      ASSERT_EQ(out.lb, src.lb) << tau;
      ASSERT_EQ(static_cast<std::size_t>(out.n_rec), keep[c].size()) << tau;
      ASSERT_EQ(wwr::wwrStreamSynchronize(out.device_stream()->stream()), wwr::wwrSuccess);
      const auto check = [&](const char *name, const auto &got, const auto &want) {
        using T = std::remove_cvref_t<decltype(want.pinned().data()[0])>;
        std::vector<T> host(keep[c].size());
        ASSERT_EQ(wwr::wwrMemcpy(host.data(), got.data(), sizeof(T) * host.size(),
                                 wwr::wwrMemcpyDeviceToHost),
                  wwr::wwrSuccess);
        for (std::size_t r = 0; r < host.size(); ++r) {
          ASSERT_EQ(host[r], want.pinned().data()[keep[c][r]])
              << name << " of class (" << out.la << ", " << out.lb << ") record " << r << ", tau "
              << tau;
        }
      };
      check("zeta", out.zeta, src.zeta);
      check("p_x", out.p_x, src.p_x);
      check("p_y", out.p_y, src.p_y);
      check("p_z", out.p_z, src.p_z);
      check("pa_x", out.pa_x, src.pa_x);
      check("pa_y", out.pa_y, src.pa_y);
      check("pa_z", out.pa_z, src.pa_z);
      check("pb_x", out.pb_x, src.pb_x);
      check("pb_y", out.pb_y, src.pb_y);
      check("pb_z", out.pb_z, src.pb_z);
      check("s", out.s, src.s);
      check("bound", out.bound, src.bound);
      check("off_a", out.off_a, src.off_a);
      check("off_b", out.off_b, src.off_b);
      // J's view leaves out what J does not read.
      const fb::PrimitivePairClassView v = out.view();
      EXPECT_EQ(v.eb, nullptr);
      EXPECT_EQ(v.bound_ab, nullptr);
    }
    EXPECT_EQ(k, pruned->size()) << tau;
    if (tau == 0.0) {
      EXPECT_EQ(pruned->size(), classes->size());
      EXPECT_EQ(record_count(*pruned), record_count(*classes));
    }
  }
  const auto none = fb::coulomb_pair_classes(res, *classes, 1e300);
  ASSERT_TRUE(none);
  EXPECT_TRUE(none->empty());
}

INSTANTIATE_TEST_SUITE_P(Molecules, CoulombPairClassesPrune,
                         ::testing::Values("h2o_sto3g", "h2o_ccpvdz", "h10_chain_ccpvdz"));

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

// The classes carry a build's scratch (J_p, D_q, the ket order), so a second
// build over them must not see the first's: one build of a density whose
// elements are reweighted (so its kets sort into another order), then two of
// the golden density on the same classes, at the default screen, each
// against libcint.
TEST(Coulomb, ReusedClassesGiveEachDensitysJ) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.vj.empty());
  double scale = 0.0;
  for (const double v : g.vj) {
    scale = std::max(scale, std::abs(v));
  }
  fb::DeviceResources res{0};
  const auto mmd = fb::primitive_pair_classes(res, g.basis);
  ASSERT_TRUE(mmd);
  auto classes = fb::coulomb_pair_classes(res, *mmd, 0.0);
  ASSERT_TRUE(classes);
  const std::size_t n = g.basis.nao_cart();
  std::vector<double> reweighted(g.density.size());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      reweighted[i * n + k] = g.density[i * n + k] * static_cast<double>(1 + i + k);
    }
  }
  fb::CartesianMatrixAO density(res, g.basis);
  fb::CartesianMatrixAO j(res, g.basis);
  upload(res, reweighted, density);
  ASSERT_TRUE(fb::coulomb(res, *classes, density, j));
  upload(res, g.density, density);
  for (const std::int32_t build : {2, 3}) {
    const fb::Status status = fb::coulomb(res, *classes, density, j);
    ASSERT_TRUE(status) << (status ? "" : status.error().message);
    const std::vector<double> got = download(res, j);
    ASSERT_EQ(got.size(), g.vj.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      ASSERT_LE(std::abs(got[i] - g.vj[i]), kRelTol * scale) << "build " << build << ", " << i;
    }
  }
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
  const auto mmd = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(mmd);
  auto classes = fb::coulomb_pair_classes(res, *mmd, 0.0);
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
  const auto mmd = fb::primitive_pair_classes(res, two);
  ASSERT_TRUE(mmd);
  auto classes = fb::coulomb_pair_classes(res, *mmd, 0.0);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO j(res, two);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

// coulomb_pair_classes refuses to make a class above d (above), so this one
// is an s class relabelled, which coulomb must still refuse.
TEST(Coulomb, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto mmd = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(mmd);
  auto classes = fb::coulomb_pair_classes(res, *mmd, 0.0);
  ASSERT_TRUE(classes);
  ASSERT_EQ(classes->size(), 1U);
  (*classes)[0].la = 3;
  (*classes)[0].lb = 3;
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
