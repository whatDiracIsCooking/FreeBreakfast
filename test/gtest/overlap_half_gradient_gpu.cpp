// GPU cases for integrals.one_electron.gradient's overlap_half_gradient: the
// overlap's derivative on the first index's centre alone, contracted per atom,
// G[B, k] = Sum_{m on B, n} A_mn dS_mn/dB_k, of the skew water in cc-pVDZ
// (s, p, d) and cc-pVTZ (to f, so every class through (ff)), for a symmetric,
// an antisymmetric and a general A, against libcint's -int1e_ipovlp contracted
// by PySCF (twice pyscf.nac.sacasscf._nac_csf;
// test/gtest/data/overlap_half_gradient/,
// tools/emit/overlap_half_gradient_reference.py); for the symmetric A, 2 G
// against overlap_gradient; and the refusals: a matrix of the wrong order and a
// basis with no per-atom map.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md). The recipe A is over PySCF's functions, so the
// matrix passed here is A_mn / (N_m N_n), whose contraction with this tree's
// derivative is PySCF's G.

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

// The issue's bar: every component to 1e-12 of max|G|, against libcint and,
// for the symmetric A, against overlap_gradient.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataFile = std::filesystem::path{FBF_TEST_DATA_DIR} /
                                        "overlap_half_gradient" / "overlap_half_gradient.json";

fb::json read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataFile);
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)["cases"]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << name << " is not a case of overlap_half_gradient.json";
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

// The recipe matrices of overlap_half_gradient_reference.py, element (m, n),
// 0-based, over PySCF's functions.
double recipe(const std::string &what, const double m, const double n) {
  if (what == "symmetric") {
    return 1.0 / (m + n + 1.0);
  }
  if (what == "antisymmetric") {
    return (m - n) / (m + n + 1.0);
  }
  return ((2.0 * m) + 1.0) / ((n + 1.0) * (m + n + 1.0));
}

class OverlapHalfGradientGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(OverlapHalfGradientGolden, MatchesLibcint) {
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
  std::vector<double> s_host(n * n);
  ASSERT_EQ(wwr::wwrMemcpyAsync(s_host.data(), s.spherical().data(), sizeof(double) * n * n,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  std::vector<double> norm(n);
  for (std::size_t i = 0; i < n; ++i) {
    norm[i] = std::sqrt(s_host[i + (n * i)]);
  }

  for (const std::string what : {"symmetric", "antisymmetric", "general"}) {
    // A over this tree's functions, column-major: element (m, n) at m + n n_ao.
    std::vector<double> a_host(n * n);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        a_host[i + (n * j)] =
            recipe(what, static_cast<double>(i), static_cast<double>(j)) / (norm[i] * norm[j]);
      }
    }
    fb::SphericalMatrixAO a(res, *basis);
    ASSERT_EQ(wwr::wwrMemcpyAsync(a.data(), a_host.data(), sizeof(double) * a_host.size(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(res.sync().has_value());

    const auto g = fb::overlap_half_gradient(res, *basis, *classes, a);
    ASSERT_TRUE(g) << g.error().message;
    std::vector<double> want;
    for (const auto &row : c["gradients"][what]) {
      for (const auto &x : row) {
        want.push_back(x.get<double>());
      }
    }
    ASSERT_EQ(g->size(), want.size()) << what;
    ASSERT_EQ(g->size(), 3 * static_cast<std::size_t>(basis->natoms())) << what;
    const double scale = max_abs(want);
    double worst = 0.0;
    for (std::size_t x = 0; x < want.size(); ++x) {
      const double err = std::abs((*g)[x] - want[x]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale)
          << name << " " << what << " G[" << x / 3 << ", " << "xyz"[x % 3] << "]";
    }
    std::cout << std::format("[ half grad  ] {} {}: max|G| {:.4f}, worst |G - libcint| / max|G| "
                             "{:.3e}\n",
                             name, what, scale, worst / scale);

    if (what == "symmetric") {
      // Both centres of a symmetric contraction give the same half.
      const auto full = fb::overlap_gradient(res, *basis, *classes, a);
      ASSERT_TRUE(full) << full.error().message;
      ASSERT_EQ(full->size(), g->size());
      const double full_scale = max_abs(*full);
      for (std::size_t x = 0; x < g->size(); ++x) {
        EXPECT_LE(std::abs((2.0 * (*g)[x]) - (*full)[x]), kRelTol * full_scale)
            << name << " 2 G vs overlap_gradient [" << x / 3 << ", " << "xyz"[x % 3] << "]";
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(H2O, OverlapHalfGradientGolden,
                         ::testing::Values("h2o_ccpvdz", "h2o_ccpvtz"));

// Each refusal before anything is launched, naming the input refused.
TEST(OverlapHalfGradient, Refusals) {
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
  fb::SphericalMatrixAO small(res, one);
  expect_refused(fb::overlap_half_gradient(res, *basis, *classes, small),
                 "overlap_half_gradient: the matrix is 1 square, but the basis has 2 spherical "
                 "AOs");

  // A basis built from shells alone has no atoms to give the gradient's rows.
  const auto one_classes = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::overlap_half_gradient(res, one, *one_classes, small),
                 "overlap_half_gradient: the basis carries no per-atom shell map");
}

} // namespace
