// GPU cases for integrals.eri.density_fitting.metric: the spherical Coulomb
// metric (P|Q) of the compiled-in cc-pVDZ-JKFIT (aux f) and cc-pVTZ-JKFIT
// (aux g) over FelizJueves's water -- the aux h and i classes the kernels
// also carry have no carried basis set and are df_high_aux_gpu.cpp's --
// against PySCF's int2c2e
// (data/density_fitting/); exactly symmetric, and bit-identical across two
// runs; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;

namespace {

namespace fb = free_breakfast;

// The issue's bar, FelizJueves's own against libcint (metric.md,
// "Tolerances"): 1e-12 of max|M|.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The worst |a_ij - a_ji| of the n x n a: 0 for an exactly symmetric one.
double asymmetry(const std::vector<double> &a, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = j + 1; i < n; ++i) {
      worst = std::max(worst, std::abs(a[i + n * j] - a[j + n * i]));
    }
  }
  return worst;
}

// The case's aux basis over its molecule, placed as the DF steps will, with
// its shells checked against the manifest's (atom, l), one contraction each;
// then the metric twice, each into a pair of its own, against int2c2e.
void check_metric(const std::string &name) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = nullptr;
  for (const fb::json &entry : (*doc)["integrals"]) {
    if (entry["name"].get<std::string>() == name) {
      c = &entry;
    }
  }
  ASSERT_NE(c, nullptr) << "no integrals case " << name;

  fb::DeviceResources res{0};
  std::vector<fb::Atom> atoms;
  for (const auto &atom : (*c)["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_aux_basis((*c)["aux_basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto aux = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;

  const auto naux = (*c)["naux"].get<std::size_t>();
  ASSERT_EQ(aux->nao_sphe(), naux);
  const auto &shells = (*c)["aux_shells"];
  ASSERT_EQ(aux->nshells(), shells.size());
  std::int32_t lmax = 0;
  for (std::uint32_t s = 0; s < aux->nshells(); ++s) {
    const std::int32_t l = aux->shells()[s].angular_momentum();
    EXPECT_EQ(l, shells[s]["l"].get<std::int32_t>()) << "aux shell " << s;
    EXPECT_EQ(shells[s]["nctr"].get<std::int32_t>(), 1) << "aux shell " << s;
    const auto atom = shells[s]["atom"].get<std::uint32_t>();
    EXPECT_GE(s, aux->atom_shell_offset(atom)) << "aux shell " << s;
    EXPECT_LT(s, aux->atom_shell_offset(atom) + aux->atom_shell_count(atom)) << "aux shell " << s;
    lmax = std::max(lmax, l);
  }
  // DZ's fitting set reaches f and TZ's g: the classes up to (f|f) and (g|g).
  EXPECT_EQ(lmax, name == "h2o_ccpvdz" ? 3 : 4);

  const auto golden = fb::read_npy(kDataDir / (*c)["int2c2e"].get<std::string>());
  ASSERT_TRUE(golden) << golden.error().message;
  ASSERT_EQ(golden->shape, (std::vector<std::size_t>{naux, naux}));
  double scale = 0.0;
  for (const double v : golden->data) {
    scale = std::max(scale, std::abs(v));
  }

  const auto u = fb::upload_spherical_transform_matrix(res, *aux);
  ASSERT_TRUE(u) << u.error().message;
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    fb::MatrixAOPair m(res, *aux, *u);
    const fb::Status status = fb::metric(res, *aux, m);
    ASSERT_TRUE(status) << status.error().message;
    const std::size_t ncart = aux->nao_cart();
    EXPECT_EQ(asymmetry(download(res, m.cartesian().data(), ncart * ncart), ncart), 0.0)
        << "the Cartesian metric";
    run = download(res, m.spherical().data(), naux * naux);
  }

  // Both column-major: element (P, Q) at P + naux Q in each.
  const std::vector<double> &spherical = runs[0];
  double worst = 0.0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < spherical.size(); ++i) {
    if (const double d = std::abs(spherical[i] - golden->data[i]); d > worst) {
      worst = d;
      at = i;
    }
  }
  EXPECT_LE(worst, kRelTol * scale) << name << ": worst |M - int2c2e| " << worst << " at ("
                                    << at % naux << ", " << at / naux << "), max|M| " << scale;
  std::cout << std::format("[ metric   ] {}: naux {}, max|M| {:.4f}, worst |M - int2c2e| / "
                           "max|M| {:.3e}\n",
                           name, naux, scale, worst / scale);

  EXPECT_EQ(asymmetry(spherical, naux), 0.0) << "the spherical metric";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

TEST(DfMetric, DzAuxThroughFMatchesInt2c2e) {
  check_metric("h2o_ccpvdz");
}

TEST(DfMetric, TzAuxThroughGMatchesInt2c2e) {
  check_metric("h2o_ccpvtz");
}

// An aux shell above i has no kernel; a pair of another basis's orders, and a
// pair on another device, are refused before anything is queued. The ladder
// reaches i (kMetricMaxMomentum), so the shell that has no kernel is k, and
// the aux h and i classes that do are run by df_high_aux_gpu.cpp.
TEST(DfMetric, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet k_shell(std::vector<fb::Shell>{fb::Shell{7, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  {
    const auto u = fb::upload_spherical_transform_matrix(res, k_shell);
    ASSERT_TRUE(u) << u.error().message;
    fb::MatrixAOPair m(res, k_shell, *u);
    const fb::Status status = fb::metric(res, k_shell, m);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(status.error().message.find("momentum 7"), std::string::npos)
        << status.error().message;
  }
  const fb::BasisSet sp(std::vector<fb::Shell>{fb::Shell{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}},
                                               fb::Shell{1, {{1.0, 1.0}}, {0.0, 0.0, 1.0}}});
  const fb::BasisSet s(std::vector<fb::Shell>{fb::Shell{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  fb::MatrixAOPair m(res, s, nullptr);
  const fb::Status status = fb::metric(res, sp, m);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(status.error().message.find("metric: the matrices are 1 x 1"), std::string::npos)
      << status.error().message;
}

} // namespace
