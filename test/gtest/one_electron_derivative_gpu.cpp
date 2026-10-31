// GPU cases for integrals.one_electron.gradient's derivative matrices: S^x, T^x
// and V^x, dX_mn/dR_Ak for every atom and axis, of the skew water in cc-pVDZ
// (s, p, d) and cc-pVTZ (to f, so every class through (ff)), against libcint's
// int1e_ipovlp, int1e_ipkin and int1e_ipnuc + int1e_iprinv assembled as
// pyscf.grad.rhf's hcore generator does, recorded by PySCF
// (test/gtest/data/one_electron_derivative/,
// tools/emit/one_electron_derivative_reference.py); each set's sum over atoms;
// each matrix contracted with the recipe matrix against the scalar gradient
// contractions (overlap_gradient and its siblings); and the refusals: a basis
// with no per-atom map, and a molecule of another atom count.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md). The normalisation does not move with the
// nuclei, so X^x_mn / (N_m N_n) here is PySCF's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// The issue's bar: every element to 1e-12 of the set's max|X^x|; the
// contraction against the scalar gradient to 1e-12 of max|G|; the sum over
// atoms below 1e-12 of max|X^x|.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "one_electron_derivative";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

fb::json read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "one_electron_derivative.json");
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)["cases"]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << name << " is not a case of one_electron_derivative.json";
  return {};
}

std::vector<fb::Atom> case_atoms(const fb::json &c) {
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  return atoms;
}

double max_abs(const std::vector<double> &v) {
  double out = 0.0;
  for (const double x : v) {
    out = std::max(out, std::abs(x));
  }
  return out;
}

// One set of matrices, ours over PySCF's functions, against the reference;
// its sum over atoms; and each matrix contracted with m against g, the
// scalar contraction of the same derivative.
void expect_matrices(const std::string &what, const std::vector<double> &got,
                     const std::vector<double> &want, const std::vector<double> &norm,
                     const std::vector<double> &m, const std::vector<double> &g) {
  const std::size_t n = norm.size();
  const std::size_t nn = n * n;
  ASSERT_EQ(got.size(), want.size()) << what;
  ASSERT_EQ(got.size(), g.size() * nn) << what;
  const double scale = max_abs(want);
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t x = 0; x < g.size(); ++x) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t e = (x * nn) + i + (n * j);
        const double err = std::abs((got[e] / (norm[i] * norm[j])) - want[e]);
        worst = std::max(worst, err);
        if (err > kRelTol * scale && bad++ < 8) {
          ADD_FAILURE() << what << " X[" << x << "](" << i << ", " << j << "): " << err;
        }
      }
    }
  }
  EXPECT_EQ(bad, 0U) << what;

  // Sum_A X^(A,k) = 0, element by element.
  double drift = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t e = 0; e < nn; ++e) {
      double sum = 0.0;
      for (std::size_t x = k; x < g.size(); x += 3) {
        sum += got[(x * nn) + e] / (norm[e % n] * norm[e / n]);
      }
      drift = std::max(drift, std::abs(sum));
    }
  }
  EXPECT_LE(drift, kRelTol * scale) << what << ": Sum_A X^(A,k)";

  // Sum_mn M_mn X^x_mn against the scalar contraction, our functions both.
  const double g_scale = max_abs(g);
  double g_worst = 0.0;
  for (std::size_t x = 0; x < g.size(); ++x) {
    double sum = 0.0;
    for (std::size_t e = 0; e < nn; ++e) {
      sum += m[e] * got[(x * nn) + e];
    }
    const double err = std::abs(sum - g[x]);
    g_worst = std::max(g_worst, err);
    EXPECT_LE(err, kRelTol * g_scale) << what << " G[" << x / 3 << ", " << "xyz"[x % 3] << "]";
  }
  ::testing::Test::RecordProperty(what + "_worst_rel_error", std::format("{:.3e}", worst / scale));
  std::cout << std::format("[ derivative ] {}: max|X| {:.4f}, worst |X - libcint| / max|X| {:.3e}, "
                           "|Sum_A X| {:.3e}, worst |M.X - G| / max|G| {:.3e}\n",
                           what, scale, worst / scale, drift / scale, g_worst / g_scale);
}

class OneElectronDerivativeGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(OneElectronDerivativeGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  const fb::json c = read_case(name);
  ASSERT_FALSE(c.is_null());
  fb::DeviceResources res{0};
  const fb::Molecule molecule(res, case_atoms(c));
  const auto stamp = fb::resolve_basis(c["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const std::size_t n = basis->nao_sphe();
  ASSERT_EQ(n, c["nao"].get<std::size_t>());
  std::int32_t lmax = 0;
  for (const fb::Shell &shell : basis->shells()) {
    lmax = std::max(lmax, shell.angular_momentum());
  }
  EXPECT_EQ(lmax, name == "h2o_ccpvdz" ? 2 : 3);
  const std::size_t n_matrix = 3 * static_cast<std::size_t>(basis->natoms());

  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto u = fb::upload_spherical_transform_matrix(res, *basis);
  ASSERT_TRUE(u) << u.error().message;
  fb::MatrixAOPair s(res, *basis, *u);
  ASSERT_TRUE(fb::overlap(res, *classes, s.cartesian()));
  ASSERT_TRUE(s.cartesian_to_spherical(res));
  const std::vector<double> s_host = download(res, s.spherical().data(), n * n);
  std::vector<double> norm(n);
  for (std::size_t i = 0; i < n; ++i) {
    norm[i] = std::sqrt(s_host[i + (n * i)]);
  }

  // A non-trivial symmetric M for the contraction check: the recipe
  // M_mn = 1 / (m + n + 1).
  std::vector<double> m_host(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m_host[i + (n * j)] = 1.0 / static_cast<double>(i + j + 1);
    }
  }
  fb::SphericalMatrixAO m(res, *basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), m_host.data(), sizeof(double) * m_host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  const auto reference = [&](const std::string &key) {
    const auto a = fb::read_npy(kDataDir / c["matrices"][key].get<std::string>());
    EXPECT_TRUE(a) << key;
    if (!a) {
      return std::vector<double>{};
    }
    EXPECT_EQ(a->shape, (std::vector<std::size_t>{n, n, n_matrix})) << key;
    return a->data;
  };

  const auto xs = fb::overlap_derivative(res, *basis, *classes);
  ASSERT_TRUE(xs) << xs.error().message;
  const auto gs = fb::overlap_gradient(res, *basis, *classes, m);
  ASSERT_TRUE(gs) << gs.error().message;
  expect_matrices(name + "_overlap", download(res, xs->data(), n_matrix * n * n),
                  reference("overlap"), norm, m_host, *gs);

  const auto xt = fb::kinetic_derivative(res, *basis, *classes);
  ASSERT_TRUE(xt) << xt.error().message;
  const auto gt = fb::kinetic_gradient(res, *basis, *classes, m);
  ASSERT_TRUE(gt) << gt.error().message;
  expect_matrices(name + "_kinetic", download(res, xt->data(), n_matrix * n * n),
                  reference("kinetic"), norm, m_host, *gt);

  const auto xv = fb::potential_derivative(res, *basis, *classes, molecule);
  ASSERT_TRUE(xv) << xv.error().message;
  const auto gv = fb::potential_gradient(res, *basis, *classes, molecule, m);
  ASSERT_TRUE(gv) << gv.error().message;
  expect_matrices(name + "_potential", download(res, xv->data(), n_matrix * n * n),
                  reference("potential"), norm, m_host, *gv);
}

INSTANTIATE_TEST_SUITE_P(H2O, OneElectronDerivativeGolden,
                         ::testing::Values("h2o_ccpvdz", "h2o_ccpvtz"));

// Each refusal before anything is launched, naming the input refused.
TEST(OneElectronDerivative, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(
      res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(1)},
            fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 1.4}, static_cast<fb::Element>(1)}});
  const auto stamp = fb::resolve_basis("sto-3g");
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto expect_refused = [](const auto &got, const std::string &says) {
    ASSERT_FALSE(got) << says;
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << got.error().message;
    EXPECT_NE(got.error().message.find(says), std::string::npos) << got.error().message;
  };

  // A basis built from shells alone has no atoms to give the matrices.
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto one_classes = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::overlap_derivative(res, one, *one_classes),
                 "overlap_derivative: the basis carries no per-atom shell map");
  expect_refused(fb::kinetic_derivative(res, one, *one_classes),
                 "kinetic_derivative: the basis carries no per-atom shell map");

  // The nuclei of a molecule the basis was not placed on.
  const fb::Molecule lone(
      res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(1)}});
  expect_refused(fb::potential_derivative(res, *basis, *classes, lone),
                 "potential_derivative: the molecule has 1 atoms, but the basis 2");
}

} // namespace
