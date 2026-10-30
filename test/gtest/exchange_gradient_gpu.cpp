// GPU cases for exchange_gradient, the exchange energy's nuclear gradient
// dE_K/dR for E_K = 1/2 Sum D_ab D_cd (ac|bd), over FelizJueves's gk kernels:
// the skew water in cc-pVDZ (s, p, d) and cc-pVTZ (to f, so every K class
// through (ff|ff)) against libcint's int2e_ip1 contracted with the recipe
// matrix M_mn = 1 / (m + n + 1) as D, recorded by PySCF
// (test/gtest/data/gradient/, tools/emit/gradient_reference.py); each one's sum
// over atoms; and the refusals: a density of the wrong order and a basis with
// no per-atom map.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md, as one_electron_gradient_gpu.cpp compares). The
// normalisation does not move with the nuclei, so D_mn / (N_m N_n) here gives
// PySCF's G.

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

// The issue's bar: 1e-10 of max|G|, per component; and the sum over atoms
// below 1e-10 absolute.
constexpr double kRelTol = 1e-10;
constexpr double kSumTol = 1e-10;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The case's "integrals" entry of gradient.json.
fb::json read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "gradient.json");
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)["integrals"]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << name << " is not an integrals case of gradient.json";
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

// A reference gradient's [x, y, z] rows, flat and row-major.
std::vector<double> flat_rows(const fb::json &rows) {
  std::vector<double> flat;
  for (const auto &row : rows) {
    for (const double v : row.get<std::vector<double>>()) {
      flat.push_back(v);
    }
  }
  return flat;
}

// The case's dE_K/dR against libcint's: to kRelTol of max|reference| per
// component, and each axis's sum over atoms below kSumTol.
void expect_matches_libcint(const std::string &name) {
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

  const auto mmd = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(mmd) << mmd.error().message;
  auto classes = fb::ordered_pair_classes(res, *basis, *mmd);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto u = fb::upload_spherical_transform_matrix(res, *basis);
  ASSERT_TRUE(u) << u.error().message;
  fb::MatrixAOPair s(res, *basis, *u);
  ASSERT_TRUE(fb::overlap(res, *mmd, s.cartesian()));
  ASSERT_TRUE(s.cartesian_to_spherical(res));
  const std::vector<double> s_host = download(res, s.spherical().data(), n * n);

  // The recipe D_mn = 1 / (m + n + 1) over PySCF's functions, over ours.
  // Symmetric, so the column-major upload is the row-major one.
  std::vector<double> d_host(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      d_host[i + n * j] =
          (1.0 / static_cast<double>(i + j + 1)) / std::sqrt(s_host[i + n * i] * s_host[j + n * j]);
    }
  }
  fb::SphericalMatrixAO d(res, *basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), d_host.data(), sizeof(double) * d_host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  const auto start = std::chrono::steady_clock::now();
  const auto got = fb::exchange_gradient(res, *basis, *classes, d);
  const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(got) << got.error().message;
  const std::vector<double> want = flat_rows(c["exchange"]);
  ASSERT_EQ(got->size(), want.size());
  ASSERT_EQ(got->size(), 3 * basis->natoms());
  double scale = 0.0;
  for (const double v : want) {
    scale = std::max(scale, std::abs(v));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got->size(); ++i) {
    const double err = std::abs((*got)[i] - want[i]);
    worst = std::max(worst, err);
    EXPECT_LE(err, kRelTol * scale) << name << " G[" << i / 3 << ", " << "xyz"[i % 3] << "]";
  }
  double worst_sum = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    double sum = 0.0;
    for (std::size_t a = k; a < got->size(); a += 3) {
      sum += (*got)[a];
    }
    worst_sum = std::max(worst_sum, std::abs(sum));
    EXPECT_LE(std::abs(sum), kSumTol) << name << ": Sum_A G[A, " << "xyz"[k] << "]";
  }
  ::testing::Test::RecordProperty(name + "_worst_rel_error", std::format("{:.3e}", worst / scale));
  std::cout << std::format("[ gradient ] {}_exchange: max|G| {:.4f}, worst |G - libcint| / max|G| "
                           "{:.3e}, worst |Sum_A G| {:.3e}, {:.2f} s\n",
                           name, scale, worst / scale, worst_sum, took.count());
}

TEST(ExchangeGradientGolden, H2OCcPvdz) {
  FBF_REQUIRE_GPU();
  expect_matches_libcint("h2o_ccpvdz");
}

// Every K class through (ff|ff), unscreened: about 5 s on the RX 9060 XT,
// under slow_test.h's ten, so it stays in the fast tier.
TEST(ExchangeGradientGolden, H2OCcPvtz) {
  FBF_REQUIRE_GPU();
  expect_matches_libcint("h2o_ccpvtz");
}

// Each refusal before anything is launched, naming the input refused.
TEST(ExchangeGradient, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(
      res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(1)},
            fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 1.4}, static_cast<fb::Element>(1)}});
  const auto stamp = fb::resolve_basis("sto-3g");
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto mmd = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(mmd) << mmd.error().message;
  auto classes = fb::ordered_pair_classes(res, *basis, *mmd);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto expect_refused = [](const auto &got, const std::string &says) {
    ASSERT_FALSE(got) << says;
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << got.error().message;
    EXPECT_NE(got.error().message.find(says), std::string::npos) << got.error().message;
  };

  // A density of another basis's order.
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::SphericalMatrixAO wrong(res, one);
  expect_refused(fb::exchange_gradient(res, *basis, *classes, wrong),
                 "exchange_gradient: the density is 1 square, but the basis has 2 spherical AOs");

  // A basis built from shells alone has no atoms to give the gradient rows.
  const auto one_mmd = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_mmd) << one_mmd.error().message;
  auto one_classes = fb::ordered_pair_classes(res, one, *one_mmd);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::exchange_gradient(res, one, *one_classes, wrong),
                 "exchange_gradient: the basis carries no per-atom shell map");

  // Classes of another basis: exchange's own check, on the Cartesian order.
  const fb::SphericalMatrixAO d(res, *basis);
  expect_refused(fb::exchange_gradient(res, *basis, *one_classes, d),
                 "is of a basis of 1 Cartesian AOs, but the density is 2 x 2");
}

} // namespace
