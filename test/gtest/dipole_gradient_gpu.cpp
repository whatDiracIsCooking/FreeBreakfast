// GPU cases for integrals.one_electron.dipole_gradient: the dipole integrals'
// nuclear derivative contracted with the recipe matrix M_mn = 1 / (m + n + 1),
// G[c, A, k] = Sum_mn M_mn d mu^c_mn / dR_Ak, of the skew water in cc-pVDZ
// (s, p, d) and cc-pVTZ (to f, so every class through (ff)), against libcint's
// int1e_irp contracted by PySCF (test/gtest/data/dipole_derivative/,
// tools/emit/dipole_derivative_reference.py); each component's sum over atoms
// against delta_ck Sum_mn M_mn S_mn; and the refusals: a matrix of the wrong
// order and a basis with no per-atom map.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md). The normalisation does not move with the
// nuclei, so M_mn / (N_m N_n) here gives PySCF's G.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// Against libcint, of max |G|, per element; FelizJueves holds its host
// reference to int1e_irp at 1.7e-15 of max |G|.
constexpr double kRelTol = 1e-10;
// Sum_A G[c, A, k] - delta_ck tr(M S), of max |G|.
constexpr double kSumRelTol = 1e-12;

const std::filesystem::path kDataDir =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "dipole_derivative";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The case's "integrals" entry of dipole_derivative.json.
fb::json read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "dipole_derivative.json");
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)["integrals"]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << name << " is not an integrals case of dipole_derivative.json";
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

class DipoleGradientGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(DipoleGradientGolden, MatchesLibcint) {
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

  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto u = fb::upload_spherical_transform_matrix(res, *basis);
  ASSERT_TRUE(u) << u.error().message;
  fb::MatrixAOPair s(res, *basis, *u);
  ASSERT_TRUE(fb::overlap(res, *classes, s.cartesian()));
  ASSERT_TRUE(s.cartesian_to_spherical(res));
  const std::vector<double> s_host = download(res, s.spherical().data(), n * n);

  // The recipe M_mn = 1 / (m + n + 1) over PySCF's functions, over ours.
  // Symmetric, so the column-major upload is the row-major one.
  std::vector<double> m_host(n * n);
  double trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m_host[i + (n * j)] = (1.0 / static_cast<double>(i + j + 1)) /
                            std::sqrt(s_host[i + (n * i)] * s_host[j + (n * j)]);
      trace += m_host[i + (n * j)] * s_host[i + (n * j)];
    }
  }
  fb::SphericalMatrixAO m(res, *basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), m_host.data(), sizeof(double) * m_host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  const auto got = fb::dipole_gradient(res, *basis, *classes, m);
  ASSERT_TRUE(got) << got.error().message;
  const std::size_t n_atom = basis->natoms();
  ASSERT_EQ(got->size(), 9 * n_atom);
  std::vector<double> want;
  for (const auto &component : c["gradient"]) {
    for (const auto &row : component) {
      for (const double v : row.get<std::vector<double>>()) {
        want.push_back(v);
      }
    }
  }
  ASSERT_EQ(want.size(), got->size());
  const double scale = std::ranges::fold_left(
      want, 0.0, [](double a, double v) { return std::max(a, std::abs(v)); });
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double err = std::abs((*got)[i] - want[i]);
    worst = std::max(worst, err);
    EXPECT_LE(err, kRelTol * scale) << "G[" << "xyz"[i / (3 * n_atom)] << ", " << (i / 3) % n_atom
                                    << ", " << "xyz"[i % 3] << "]";
  }
  double drift = 0.0;
  for (std::size_t comp = 0; comp < 3; ++comp) {
    for (std::size_t k = 0; k < 3; ++k) {
      double sum = comp == k ? -trace : 0.0;
      for (std::size_t a = 0; a < n_atom; ++a) {
        sum += (*got)[(3 * ((n_atom * comp) + a)) + k];
      }
      drift = std::max(drift, std::abs(sum));
    }
  }
  EXPECT_LE(drift, kSumRelTol * scale);
  std::cout << std::format("[ dipole_g ] {}: max |G| {:.4f}, |G - libcint| / max |G| {:.3e}, "
                           "sum rule {:.2e}\n",
                           name, scale, worst / scale, drift);
}

INSTANTIATE_TEST_SUITE_P(H2O, DipoleGradientGolden, ::testing::Values("h2o_ccpvdz", "h2o_ccpvtz"));

// Each refusal before anything is launched, naming the input refused.
TEST(DipoleGradient, Refusals) {
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

  // A matrix of another basis's order.
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::SphericalMatrixAO wrong(res, one);
  expect_refused(fb::dipole_gradient(res, *basis, *classes, wrong),
                 "dipole_gradient: the matrix is 1 square, but the basis has 2 spherical AOs");

  // A basis built from shells alone has no atoms to give the gradient rows.
  const auto one_classes = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::dipole_gradient(res, one, *one_classes, wrong),
                 "dipole_gradient: the basis carries no per-atom shell map");
}

} // namespace
