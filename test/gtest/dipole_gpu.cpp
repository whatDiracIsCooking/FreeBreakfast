// GPU cases for integrals.one_electron.dipole: the dipole matrices of H2O in
// STO-3G (s, p) and cc-pVDZ (s, p, d), built Cartesian and carried to the
// spherical layout, against libcint's int1e_r_sph at origin 0, recorded by
// PySCF (test/gtest/data/cis/, tools/emit/cis_reference.py); plus the zeroing,
// the refusal of an f shell, and of three matrices that differ in order.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md, as df_three_center_gpu.cpp compares). The BSE
// coefficients' printed digits leave N_i - 1 up to ~1e-10 in STO-3G and ~1e-6
// in cc-pVDZ, so mu_ij is divided by N_i N_j before it is compared.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.basis_set;

namespace {

namespace fb = free_breakfast;

// The one-electron families' bar against libcint (overlap_gpu.cpp), here
// absolute: H2O's |mu| is below 2 bohr, so it is the tighter of the two.
constexpr double kTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "cis";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The case's molecule from cis.json, its basis by name as the Basis step reads it.
fb::BasisSet case_basis(fb::DeviceResources &res, const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "cis.json");
  EXPECT_TRUE(doc) << doc.error().message;
  if (!doc) {
    return fb::BasisSet(std::vector<fb::Shell>{});
  }
  for (const fb::json &c : (*doc)["cases"]) {
    if (c["name"].get<std::string>() != name) {
      continue;
    }
    std::vector<fb::Atom> atoms;
    for (const auto &atom : c["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                               static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
    }
    const fb::MoleculeOutput molecule{fb::Molecule(res, std::move(atoms))};
    auto basis = fb::BasisSetStep{.basis = c["basis"].get<std::string>()}.run(
        fb::View<fb::MoleculeOutput>{molecule});
    EXPECT_TRUE(basis) << basis.error().message;
    if (!basis) {
      return fb::BasisSet(std::vector<fb::Shell>{});
    }
    EXPECT_EQ(basis->basis.nao_sphe(), c["nao"].get<std::uint32_t>()) << name;
    return std::move(basis->basis);
  }
  ADD_FAILURE() << name << " is not a case of cis.json";
  return fb::BasisSet(std::vector<fb::Shell>{});
}

class DipoleGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(DipoleGolden, SphericalMatchesLibcint) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const std::string name = GetParam();
  const fb::BasisSet basis = case_basis(res, name);
  const std::size_t n = basis.nao_sphe();
  ASSERT_GT(n, 0U);
  // (3, n, n), read back column-major: (d, m, n') at d + 3 (m + n n').
  const auto golden = fb::read_npy(kDataDir / (name + "_int1e_r.npy"));
  ASSERT_TRUE(golden) << golden.error().message;
  ASSERT_EQ(golden->shape, (std::vector<std::size_t>{3, n, n}));

  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto u = fb::upload_spherical_transform_matrix(res, basis);
  ASSERT_TRUE(u) << u.error().message;
  std::array<fb::MatrixAOPair, 3> mu{fb::MatrixAOPair(res, basis, *u),
                                     fb::MatrixAOPair(res, basis, *u),
                                     fb::MatrixAOPair(res, basis, *u)};
  const fb::Status built =
      fb::dipole(res, *classes, mu[0].cartesian(), mu[1].cartesian(), mu[2].cartesian());
  ASSERT_TRUE(built) << built.error().message;

  fb::MatrixAOPair s(res, basis, *u);
  ASSERT_TRUE(fb::overlap(res, *classes, s.cartesian()));
  ASSERT_TRUE(s.cartesian_to_spherical(res));
  const std::vector<double> s_host = download(res, s.spherical().data(), n * n);
  std::vector<double> norm(n);
  for (std::size_t i = 0; i < n; ++i) {
    norm[i] = std::sqrt(s_host[i + n * i]);
  }

  double worst = 0.0;
  for (std::size_t d = 0; d < 3; ++d) {
    ASSERT_TRUE(mu[d].cartesian_to_spherical(res));
    const std::vector<double> got =
        download(res, mu[d].spherical().data(), mu[d].spherical().num_elements());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double ours = got[i + n * j] / (norm[i] * norm[j]);
        const double err = std::abs(ours - golden->data[d + 3 * (i + n * j)]);
        worst = std::max(worst, err);
        EXPECT_LE(err, kTol) << "mu_" << "xyz"[d] << "(" << i << ", " << j << ")";
      }
    }
  }
  RecordProperty("worst_abs_error", std::format("{:.3e}", worst));
  std::cout << std::format("[ dipole   ] {}: worst |mu - int1e_r_sph| {:.3e}\n", name, worst);
}

INSTANTIATE_TEST_SUITE_P(H2O, DipoleGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

TEST(Dipole, ZeroesTheMatricesFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  std::array<fb::CartesianMatrixAO, 3> mu{fb::CartesianMatrixAO(res, basis),
                                          fb::CartesianMatrixAO(res, basis),
                                          fb::CartesianMatrixAO(res, basis)};
  const double stale = 42.0;
  for (fb::CartesianMatrixAO &m : mu) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                  res.stream()),
              wwr::wwrSuccess);
  }
  // No classes: the only work left is the zeroing.
  ASSERT_TRUE(fb::dipole(res, {}, mu[0], mu[1], mu[2]));
  for (const fb::CartesianMatrixAO &m : mu) {
    EXPECT_EQ(download(res, m.data(), 1)[0], 0.0);
  }
}

TEST(Dipole, RefusesAnFShell) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const fb::BasisSet basis(
      {fb::Shell(0, {{1.0, 1.0}}, origin), fb::Shell(fb::kDipoleMaxL + 1, {{1.0, 1.0}}, origin)});
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  std::array<fb::CartesianMatrixAO, 3> mu{fb::CartesianMatrixAO(res, basis),
                                          fb::CartesianMatrixAO(res, basis),
                                          fb::CartesianMatrixAO(res, basis)};
  const fb::Status status = fb::dipole(res, *classes, mu[0], mu[1], mu[2]);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(status.error().message.find("f shell"), std::string::npos) << status.error().message;
}

TEST(Dipole, RefusesMatricesOfDifferentOrders) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, origin)});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, origin), fb::Shell(0, {{0.5, 1.0}}, origin)});
  fb::CartesianMatrixAO mu_x(res, one);
  fb::CartesianMatrixAO mu_y(res, one);
  fb::CartesianMatrixAO mu_z(res, two);
  const fb::Status status = fb::dipole(res, {}, mu_x, mu_y, mu_z);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
