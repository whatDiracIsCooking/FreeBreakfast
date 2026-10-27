// GPU cases for the top of the density-fitting aux ladder: the metric (P|Q)
// through (i|i) and the three-centre (mn|P) through (ff|i), over the
// hand-built basis of data/df_high_aux/ -- one shell per l on two skew
// centres, orbitals to f and aux to i -- against PySCF's int2c2e and
// int3c2e.
//
// WHY A HAND-BUILT BASIS. df_metric_gpu.cpp and df_three_center_gpu.cpp run
// the two fitting sets this tree carries, whose aux shells stop at g: between
// them they reach (g|g) and (ff|g) and nothing above. The emitted ladder goes
// to (i|i) and (ff|i) (kMetricMaxMomentum and kThreeCenterAuxMaxMomentum),
// and no carried set has an i fitting shell, so the classes above g have no
// other way to be run. This fixture is the smallest basis that reaches every
// one of them, which the coverage case below checks rather than asserts.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting;

namespace {

namespace fb = free_breakfast;

// The same bar the carried sets hold to, 1e-12 of max (df_metric_gpu.cpp,
// df_three_center_gpu.cpp), and these classes clear it by two orders: at
// L = 12 the measured worst is 1.2e-14 of max|M| for the metric and 1.6e-14
// of max|(mn|P)| for the three-centre, on the RTX 3080. The tower does not
// visibly lose accuracy up the aux ladder, so there is no reason to grant it
// a looser tolerance than the g classes get. Each case prints what it
// achieved, so a regression that still passes shows up as a number.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "df_high_aux";

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The manifest's shells, built exactly as it holds them: one primitive each,
// its coefficient already the primitive's own normaliser, so these go into a
// Shell as given (as data/README.md's one_electron shells do) and need no
// primitive_normalized.
fb::BasisSet basis_of(const fb::json &shells) {
  std::vector<fb::Shell> built;
  for (const fb::json &shell : shells) {
    const auto centre = shell["centre"].get<std::vector<double>>();
    built.push_back(
        fb::Shell{shell["l"].get<std::int32_t>(),
                  {{shell["exponent"].get<double>(), shell["coefficient"].get<double>()}},
                  {centre[0], centre[1], centre[2]}});
  }
  return fb::BasisSet{std::move(built)};
}

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too (df_three_center_gpu.cpp's, over one primitive):
// Integral x^2l exp(-g r^2) = (2l-1)!! pi^(3/2) / ((2g)^l g^(3/2)).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  const fb::Primitive &p = shell.primitives().front();
  const double g = 2.0 * p.exponent;
  return p.coefficient * p.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
         std::pow(2.0 * g, l);
}

// Two shells on the same centre: the bra pair's P - A and P - B are then
// zero and the Boys argument 0, a different arithmetic path from a pair with
// a real separation, so the coverage case wants both for every class.
bool same_centre(const fb::Shell &a, const fb::Shell &b) {
  return a.center().x == b.center().x && a.center().y == b.center().y &&
         a.center().z == b.center().z;
}

std::int32_t max_momentum(const fb::BasisSet &basis) {
  std::int32_t l = 0;
  for (const fb::Shell &shell : basis.shells()) {
    l = std::max(l, shell.angular_momentum());
  }
  return l;
}

struct Fixture {
  fb::BasisSet basis;
  fb::BasisSet aux;
  std::size_t nao;
  std::size_t naux;
};

// Both bases as the manifest holds them, with the counts it records. Read per
// case rather than once into a static: this code does not throw
// (docs/style.md), so the read's failure belongs in a case's ASSERT.
Fixture fixture(const fb::json &doc) {
  return Fixture{basis_of(doc["orbital_shells"]), basis_of(doc["aux_shells"]),
                 doc["nao"].get<std::size_t>(), doc["naux"].get<std::size_t>()};
}

// The golden (mn|P), gathered from the manifest's aux pieces into the call's
// layout: P + naux (n + nao m), P fastest. Each piece is (nao, nao, slab) in
// Fortran order, df_reference.py's split (df_three_center_gpu.cpp gathers the
// same shape).
std::vector<double> golden_eri3(const fb::json &doc, const std::size_t nao,
                                const std::size_t naux) {
  std::vector<double> golden(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
  std::size_t covered = 0;
  for (const fb::json &piece : doc["int3c2e"]) {
    const auto range = piece["aux"].get<std::vector<std::size_t>>();
    const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
    EXPECT_TRUE(slab) << slab.error().message;
    EXPECT_EQ(slab->shape, (std::vector<std::size_t>{nao, nao, range[1] - range[0]}));
    EXPECT_EQ(range[0], covered);
    for (std::size_t p = range[0]; p < range[1]; ++p) {
      for (std::size_t n = 0; n < nao; ++n) {
        for (std::size_t m = 0; m < nao; ++m) {
          golden[p + (naux * (n + (nao * m)))] =
              slab->data[m + (nao * (n + (nao * (p - range[0]))))];
        }
      }
    }
    covered = range[1];
  }
  EXPECT_EQ(covered, naux);
  return golden;
}

double max_abs(const std::vector<double> &a) {
  double scale = 0.0;
  for (const double v : a) {
    scale = std::max(scale, std::abs(v));
  }
  return scale;
}

// The fixture is what it claims: the two momenta the ladder's limits name,
// every function unit-normalised (which is what lets the arrays be compared
// to libcint's with no norm factor), and -- the point of the file -- every
// emitted class of both families reached, each by a one-centre pair and a
// two-centre one.
TEST(DfHighAux, FixtureCoversEveryEmittedClass) {
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  EXPECT_EQ(f.basis.nao_sphe(), f.nao);
  EXPECT_EQ(f.aux.nao_sphe(), f.naux);
  // The emitted ladder's two limits -- kThreeCenterMaxMomentum (f) and
  // kMetricMaxMomentum = kThreeCenterAuxMaxMomentum (i) -- as literals, not
  // as the bridge headers' constants: those headers carry the GPU runtime's
  // includes, which have no business in a module-scanned test TU. Raising
  // either limit means regenerating this fixture (its emitter's docstring
  // says so) and lifting the literal here.
  const auto orbital_max_l = (*doc)["orbital_max_l"].get<std::int32_t>();
  const auto aux_max_l = (*doc)["aux_max_l"].get<std::int32_t>();
  EXPECT_EQ(orbital_max_l, 3);
  EXPECT_EQ(aux_max_l, 6);
  EXPECT_EQ(max_momentum(f.basis), orbital_max_l);
  EXPECT_EQ(max_momentum(f.aux), aux_max_l);

  for (const fb::BasisSet *b : {&f.basis, &f.aux}) {
    for (const fb::Shell &shell : b->shells()) {
      EXPECT_NEAR(self_overlap(shell), 1.0, 1e-13) << "shell l " << shell.angular_momentum();
    }
  }

  // Metric: every (lP, lQ) with lP <= lQ <= kMetricMaxMomentum, and whether
  // the pair's two shells share a centre.
  std::set<std::array<std::int32_t, 3>> metric_seen;
  for (std::uint32_t q = 0; q < f.aux.nshells(); ++q) {
    for (std::uint32_t p = 0; p <= q; ++p) {
      const std::int32_t lp = f.aux.shells()[p].angular_momentum();
      const std::int32_t lq = f.aux.shells()[q].angular_momentum();
      const bool same = same_centre(f.aux.shells()[p], f.aux.shells()[q]);
      metric_seen.insert({std::min(lp, lq), std::max(lp, lq), same ? 1 : 0});
    }
  }
  const auto aux_ls = std::views::iota(0, aux_max_l + 1);
  for (const std::int32_t lp : aux_ls) {
    for (const std::int32_t lq : aux_ls) {
      if (lp > lq) {
        continue;
      }
      for (const std::int32_t same : {0, 1}) {
        EXPECT_TRUE(metric_seen.contains({lp, lq, same}))
            << "no " << (same == 1 ? "one-centre" : "two-centre") << " metric pair (" << lp << "|"
            << lq << ")";
      }
    }
  }

  // eri3: every (la, lb, lP), la <= lb <= kThreeCenterMaxMomentum against
  // lP <= kThreeCenterAuxMaxMomentum, by the bra pair's centres.
  std::set<std::array<std::int32_t, 4>> eri3_seen;
  for (std::uint32_t j = 0; j < f.basis.nshells(); ++j) {
    for (std::uint32_t i = 0; i <= j; ++i) {
      const std::int32_t la = f.basis.shells()[i].angular_momentum();
      const std::int32_t lb = f.basis.shells()[j].angular_momentum();
      const bool same = same_centre(f.basis.shells()[i], f.basis.shells()[j]);
      for (const fb::Shell &aux_shell : f.aux.shells()) {
        eri3_seen.insert(
            {std::min(la, lb), std::max(la, lb), aux_shell.angular_momentum(), same ? 1 : 0});
      }
    }
  }
  for (const std::int32_t la : std::views::iota(0, orbital_max_l + 1)) {
    for (const std::int32_t lb : std::views::iota(la, orbital_max_l + 1)) {
      for (const std::int32_t lp : aux_ls) {
        for (const std::int32_t same : {0, 1}) {
          EXPECT_TRUE(eri3_seen.contains({la, lb, lp, same}))
              << "no " << (same == 1 ? "one-centre" : "two-centre") << " eri3 class (" << la << lb
              << "|" << lp << ")";
        }
      }
    }
  }
}

// (P|Q) over aux s..i: all 28 classes, (i|i) at L = 12 the top one.
TEST(DfHighAux, MetricThroughIiMatchesInt2c2e) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  fb::DeviceResources res{0};

  const auto golden = fb::read_npy(kDataDir / (*doc)["int2c2e"].get<std::string>());
  ASSERT_TRUE(golden) << golden.error().message;
  ASSERT_EQ(golden->shape, (std::vector<std::size_t>{f.naux, f.naux}));
  const double scale = max_abs(golden->data);

  const auto u = fb::upload_spherical_transform_matrix(res, f.aux);
  ASSERT_TRUE(u) << u.error().message;
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    fb::MatrixAOPair m(res, f.aux, *u);
    const fb::Status status = fb::metric(res, f.aux, m);
    ASSERT_TRUE(status) << status.error().message;
    run = download(res, m.spherical().data(), f.naux * f.naux);
  }

  double worst = 0.0;
  std::size_t at = 0;
  double asymmetry = 0.0;
  for (std::size_t q = 0; q < f.naux; ++q) {
    for (std::size_t p = 0; p < f.naux; ++p) {
      const std::size_t i = p + (f.naux * q);
      if (const double d = std::abs(runs[0][i] - golden->data[i]); d > worst) {
        worst = d;
        at = i;
      }
      asymmetry = std::max(asymmetry, std::abs(runs[0][i] - runs[0][q + (f.naux * p)]));
    }
  }
  EXPECT_LE(worst, kRelTol * scale) << "worst |M - int2c2e| " << worst << " at (" << at % f.naux
                                    << ", " << at / f.naux << "), max|M| " << scale;
  std::cout << std::format("[ metric   ] high aux: naux {}, max|M| {:.4f}, worst "
                           "|M - int2c2e| / max|M| {:.3e}\n",
                           f.naux, scale, worst / scale);
  EXPECT_EQ(asymmetry, 0.0) << "the spherical metric is not exactly symmetric";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

// (mn|P) over orbitals s..f and aux s..i: all 70 classes, (ff|i) at L = 12.
TEST(DfHighAux, ThreeCenterThroughFfIMatchesInt3c2e) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  fb::DeviceResources res{0};

  const std::vector<double> golden = golden_eri3(*doc, f.nao, f.naux);
  const double scale = max_abs(golden);
  ASSERT_GT(scale, 0.0);

  const auto classes = fb::primitive_pair_classes(res, f.basis);
  ASSERT_TRUE(classes) << classes.error().message;
  fb::Buffers::device<double> b(f.naux * f.nao * f.nao, res.device());
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    const fb::Status status = fb::three_center(res, f.basis, *classes, f.aux, b.data());
    ASSERT_TRUE(status) << status.error().message;
    run = download(res, b.data(), b.num_elements());
  }

  double worst = 0.0;
  std::array<std::size_t, 3> at{};
  double asymmetry = 0.0;
  for (std::size_t m = 0; m < f.nao; ++m) {
    for (std::size_t n = 0; n < f.nao; ++n) {
      for (std::size_t p = 0; p < f.naux; ++p) {
        const std::size_t i = p + (f.naux * (n + (f.nao * m)));
        if (const double d = std::abs(runs[0][i] - golden[i]); d > worst) {
          worst = d;
          at = {m, n, p};
        }
        asymmetry =
            std::max(asymmetry, std::abs(runs[0][i] - runs[0][p + (f.naux * (m + (f.nao * n)))]));
      }
    }
  }
  EXPECT_LE(worst, kRelTol * scale)
      << "worst |(mn|P) - int3c2e| " << worst << " at (" << at[0] << ", " << at[1] << " | " << at[2]
      << "), max|(mn|P)| " << scale;
  std::cout << std::format("[ eri3     ] high aux: nao {}, naux {}, max|(mn|P)| {:.4f}, worst "
                           "|(mn|P) - int3c2e| / max|(mn|P)| {:.3e}\n",
                           f.nao, f.naux, scale, worst / scale);
  EXPECT_EQ(asymmetry, 0.0) << "(mn|P) and (nm|P) differ";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

} // namespace
