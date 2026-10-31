// GPU cases for integrals.one_electron.hessian: the overlap, kinetic and
// nuclear-attraction Hessian contractions H[A, k, B, l] = Sum_mn M_mn
// d2X_mn/dR_Ak dR_Bl of the skew water in cc-pVDZ (s, p, d) and cc-pVTZ (to f,
// so every class through (ff)), against libcint's int1e_ipipovlp/ipovlpip,
// int1e_ipipkin/ipkinip and int1e_ipipnuc/ipnucip + the rinv terms, assembled
// as PySCF's partial_hess_elec does and contracted with the recipe matrix
// M_mn = 1 / (m + n + 1) (test/gtest/data/hessian/,
// tools/emit/one_electron_hessian_reference.py); each one's translational sum
// rule over the column atom; nuclear_repulsion_hessian against PySCF's hess_nuc; and the refusals: a
// matrix of the wrong order, a basis with no per-atom map, and a molecule of
// another atom count.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md, as one_electron_gradient_gpu.cpp compares). The
// normalisation does not move with the nuclei, so M_mn / (N_m N_n) here gives
// PySCF's H.

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

// The issue's bar: 1e-11 of max|H|, per element; and each sum over one atom
// index below 1e-11 of max|H| too.
constexpr double kRelTol = 1e-11;
constexpr double kSumRelTol = 1e-11;
// The nuclear repulsion Hessian: closed form against closed form.
constexpr double kNuclearRelTol = 1e-13;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "hessian";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The case's "integrals" entry of one_electron.json.
fb::json read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "one_electron.json");
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)["integrals"]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << name << " is not an integrals case of one_electron.json";
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

// A reference Hessian's rows, flat and row-major.
std::vector<double> flat_rows(const fb::json &rows) {
  std::vector<double> flat;
  for (const auto &row : rows) {
    for (const double v : row.get<std::vector<double>>()) {
      flat.push_back(v);
    }
  }
  return flat;
}

// Ours against the reference to rel_tol of max|reference|, and every
// Sum_B H[row, B, l] below sum_rel_tol of it. The sum over the column atom: the
// build forms each diagonal block from its column, H[A, k, A, l] = -Sum_{B !=
// A} H[B, k, A, l], so the sum over the row atom vanishes by construction,
// while this one holds only if the computed off-diagonal blocks are each
// other's transposes, (A, B) the kernels' AB and (B, A) their BA.
void expect_hessian(const std::string &what, const std::vector<double> &got,
                    const std::vector<double> &want, const double rel_tol,
                    const double sum_rel_tol) {
  ASSERT_EQ(got.size(), want.size()) << what;
  const auto n = static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(got.size()))));
  ASSERT_EQ(n * n, got.size()) << what;
  double scale = 0.0;
  for (const double v : want) {
    scale = std::max(scale, std::abs(v));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double err = std::abs(got[i] - want[i]);
    worst = std::max(worst, err);
    EXPECT_LE(err, rel_tol * scale) << what << " H[" << i / n << ", " << i % n << "]";
  }
  double drift = 0.0;
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t l = 0; l < 3; ++l) {
      double sum = 0.0;
      for (std::size_t col = l; col < n; col += 3) {
        sum += got[(row * n) + col];
      }
      drift = std::max(drift, std::abs(sum));
      EXPECT_LE(std::abs(sum), sum_rel_tol * scale)
          << what << ": Sum_B H[" << row << ", B, " << "xyz"[l] << "]";
    }
  }
  ::testing::Test::RecordProperty(what + "_worst_rel_error", std::format("{:.3e}", worst / scale));
  std::cout << std::format("[ hessian ] {}: max|H| {:.4f}, worst |H - PySCF| / max|H| {:.3e}, "
                           "worst |Sum_B H| / max|H| {:.3e}\n",
                           what, scale, worst / scale, drift / scale);
}

class OneElectronHessianGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(OneElectronHessianGolden, MatchesPySCF) {
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
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m_host[i + n * j] =
          (1.0 / static_cast<double>(i + j + 1)) / std::sqrt(s_host[i + n * i] * s_host[j + n * j]);
    }
  }
  fb::SphericalMatrixAO m(res, *basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), m_host.data(), sizeof(double) * m_host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  const auto hs = fb::overlap_hessian(res, *basis, *classes, m);
  ASSERT_TRUE(hs) << hs.error().message;
  expect_hessian(name + "_overlap", *hs, flat_rows(c["overlap"]), kRelTol, kSumRelTol);
  const auto ht = fb::kinetic_hessian(res, *basis, *classes, m);
  ASSERT_TRUE(ht) << ht.error().message;
  expect_hessian(name + "_kinetic", *ht, flat_rows(c["kinetic"]), kRelTol, kSumRelTol);
  const auto hv = fb::potential_hessian(res, *basis, *classes, molecule, m);
  ASSERT_TRUE(hv) << hv.error().message;
  expect_hessian(name + "_potential", *hv, flat_rows(c["potential"]), kRelTol, kSumRelTol);

  const std::vector<double> hn = fb::nuclear_repulsion_hessian(molecule);
  expect_hessian(name + "_nuclear_repulsion", hn, flat_rows(c["nuclear_repulsion"]), kNuclearRelTol,
                 kNuclearRelTol);
}

INSTANTIATE_TEST_SUITE_P(H2O, OneElectronHessianGolden,
                         ::testing::Values("h2o_ccpvdz", "h2o_ccpvtz"));

// Symmetric to the last bit (each T_kl is formed from r_k r_l, which commutes),
// and zero for a lone atom.
TEST(NuclearRepulsionHessian, SymmetricAndZeroForOneAtom) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule lone(
      res, {fb::Atom{fb::Vec<double, 3>{0.3, 0.0, -1.0}, static_cast<fb::Element>(8)}});
  EXPECT_EQ(fb::nuclear_repulsion_hessian(lone), std::vector<double>(9, 0.0));
  const fb::Molecule molecule(
      res, {fb::Atom{fb::Vec<double, 3>{0.1, -0.2, 0.3}, static_cast<fb::Element>(7)},
            fb::Atom{fb::Vec<double, 3>{1.9, 0.4, -0.1}, static_cast<fb::Element>(6)},
            fb::Atom{fb::Vec<double, 3>{-0.7, 1.6, 0.8}, static_cast<fb::Element>(1)}});
  const std::vector<double> h = fb::nuclear_repulsion_hessian(molecule);
  ASSERT_EQ(h.size(), 81U);
  for (std::size_t i = 0; i < 9; ++i) {
    for (std::size_t j = 0; j < 9; ++j) {
      EXPECT_EQ(h[(i * 9) + j], h[(j * 9) + i]) << i << ", " << j;
    }
  }
}

// Each refusal before anything is launched, naming the input refused.
TEST(OneElectronHessian, Refusals) {
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
  expect_refused(fb::overlap_hessian(res, *basis, *classes, wrong),
                 "overlap_hessian: the matrix is 1 square, but the basis has 2 spherical AOs");

  // A basis built from shells alone has no atoms to give the Hessian rows.
  const auto one_classes = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::kinetic_hessian(res, one, *one_classes, wrong),
                 "kinetic_hessian: the basis carries no per-atom shell map");

  // The nuclei of a molecule the basis was not placed on.
  const fb::Molecule lone(
      res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(1)}});
  const fb::SphericalMatrixAO m(res, *basis);
  expect_refused(fb::potential_hessian(res, *basis, *classes, lone, m),
                 "potential_hessian: the molecule has 1 atoms, but the basis 2");
}

} // namespace
