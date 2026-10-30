// GPU cases for integrals.eri.integral_direct.coulomb's coulomb_gradient: the
// Coulomb energy's nuclear gradient dE_J/dR_Ak, E_J = 1/2 Sum D_ab D_cd (ab|cd),
// of the skew water in cc-pVDZ (s, p, d: the raised [p|q] through L = 9) and
// cc-pVTZ (to f, so every gjab class through (ff) and the raised [p|q] at
// L = 13), against libcint's int2e_ip1 contracted with the recipe density
// D_mn = 1 / (m + n + 1), unscreened, recorded by PySCF
// (test/gtest/data/gradient/, tools/emit/gradient_reference.py); each one's sum
// over atoms; how far the energy's screen at its default moves it, measured and
// held; and the refusals: a density of the wrong order, a basis with no
// per-atom map, and a negative or NaN tau.
//
// PySCF unit-normalises each contracted function; this tree normalises only the
// primitives, so its spherical function i is N_i = sqrt(S_ii) times PySCF's
// (test/gtest/data/README.md). E_J is bilinear in D and the normalisation does
// not move with the nuclei, so D_mn / (N_m N_n) here gives PySCF's G.

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

double max_abs(const std::vector<double> &v) {
  double scale = 0.0;
  for (const double x : v) {
    scale = std::max(scale, std::abs(x));
  }
  return scale;
}

// max |got - want| / max|want|.
double worst_rel(const std::vector<double> &got, const std::vector<double> &want) {
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(got[i] - want[i]));
  }
  return worst / max_abs(want);
}

// Ours against the reference to kRelTol of max|reference|, and each axis's sum
// over atoms below kSumTol.
void expect_gradient(const std::string &what, const std::vector<double> &got,
                     const std::vector<double> &want) {
  ASSERT_EQ(got.size(), want.size()) << what;
  const double scale = max_abs(want);
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_LE(std::abs(got[i] - want[i]), kRelTol * scale)
        << what << " G[" << i / 3 << ", " << "xyz"[i % 3] << "]";
  }
  for (std::size_t k = 0; k < 3; ++k) {
    double sum = 0.0;
    for (std::size_t a = k; a < got.size(); a += 3) {
      sum += got[a];
    }
    EXPECT_LE(std::abs(sum), kSumTol) << what << ": Sum_A G[A, " << "xyz"[k] << "]";
  }
  const double worst = worst_rel(got, want);
  ::testing::Test::RecordProperty(what + "_worst_rel_error", std::format("{:.3e}", worst));
  std::cout << std::format("[ gradient ] {}: max|G| {:.4f}, worst |G - libcint| / max|G| {:.3e}\n",
                           what, scale, worst);
}

// One case's molecule, basis, classes and recipe density, on the device.
struct Case {
  fb::json c;
  fb::Molecule molecule;
  fb::BasisSet basis;
  std::vector<fb::PrimitivePairClass> classes;
  fb::SphericalMatrixAO density;
};

std::optional<Case> make_case(fb::DeviceResources &res, const std::string &name) {
  fb::json c = read_case(name);
  if (c.is_null()) {
    return std::nullopt;
  }
  fb::Molecule molecule(res, case_atoms(c));
  const auto stamp = fb::resolve_basis(c["basis"].get<std::string>());
  EXPECT_TRUE(stamp) << stamp.error().message;
  if (!stamp) {
    return std::nullopt;
  }
  auto basis = fb::place(stamp->primitive_normalized(), molecule);
  EXPECT_TRUE(basis) << basis.error().message;
  if (!basis) {
    return std::nullopt;
  }
  const std::size_t n = basis->nao_sphe();
  EXPECT_EQ(n, c["nao"].get<std::size_t>());
  auto classes = fb::primitive_pair_classes(res, *basis);
  EXPECT_TRUE(classes) << classes.error().message;
  const auto u = fb::upload_spherical_transform_matrix(res, *basis);
  EXPECT_TRUE(u) << u.error().message;
  if (!classes || !u) {
    return std::nullopt;
  }
  fb::MatrixAOPair s(res, *basis, *u);
  EXPECT_TRUE(fb::overlap(res, *classes, s.cartesian()));
  EXPECT_TRUE(s.cartesian_to_spherical(res));
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
  fb::SphericalMatrixAO density(res, *basis);
  EXPECT_EQ(wwr::wwrMemcpyAsync(density.data(), d_host.data(), sizeof(double) * d_host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return Case{std::move(c), std::move(molecule), std::move(*basis), std::move(*classes),
              std::move(density)};
}

class CoulombGradientGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(CoulombGradientGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  fb::DeviceResources res{0};
  const auto k = make_case(res, name);
  ASSERT_TRUE(k.has_value());
  std::int32_t lmax = 0;
  for (const fb::Shell &shell : k->basis.shells()) {
    lmax = std::max(lmax, shell.angular_momentum());
  }
  EXPECT_EQ(lmax, name == "h2o_ccpvdz" ? 2 : 3);
  const std::vector<double> want = flat_rows(k->c["coulomb"]);

  // The default: unscreened.
  const auto g = fb::coulomb_gradient(res, k->basis, k->classes, k->density);
  ASSERT_TRUE(g) << g.error().message;
  expect_gradient(name + "_coulomb", *g, want);

  // The energy's default screen, 1e-10, on the raised [p|q]: what it would
  // cost, measured against the same reference (README.md, "The gradient's
  // screen", has the numbers). Held to the same bar, so a screen that ever
  // moved it past the bar is seen here.
  const auto screened = fb::coulomb_gradient(res, k->basis, k->classes, k->density, 1e-10);
  ASSERT_TRUE(screened) << screened.error().message;
  const double moved = worst_rel(*screened, *g);
  ::testing::Test::RecordProperty(name + "_screen_moved", std::format("{:.3e}", moved));
  std::cout << std::format("[ gradient ] {}: tau 1e-10 moves G by {:.3e} of max|G|\n", name, moved);
  expect_gradient(name + "_coulomb_screened", *screened, want);
}

INSTANTIATE_TEST_SUITE_P(H2O, CoulombGradientGolden, ::testing::Values("h2o_ccpvdz", "h2o_ccpvtz"));

// Each refusal before anything is launched, naming the input refused.
TEST(CoulombGradient, Refusals) {
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

  // A density of another basis's order.
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::SphericalMatrixAO wrong(res, one);
  expect_refused(fb::coulomb_gradient(res, *basis, *classes, wrong),
                 "coulomb_gradient: the density is 1 square, but the basis has 2 spherical AOs");

  // A basis built from shells alone has no atoms to give the gradient rows.
  const auto one_classes = fb::primitive_pair_classes(res, one);
  ASSERT_TRUE(one_classes) << one_classes.error().message;
  expect_refused(fb::coulomb_gradient(res, one, *one_classes, wrong),
                 "coulomb_gradient: the basis carries no per-atom shell map");

  // A screen that is negative, or NaN, which would never screen.
  const fb::SphericalMatrixAO d(res, *basis);
  expect_refused(fb::coulomb_gradient(res, *basis, *classes, d, -1.0),
                 "coulomb_gradient: the screening threshold tau is -1");
  expect_refused(
      fb::coulomb_gradient(res, *basis, *classes, d, std::numeric_limits<double>::quiet_NaN()),
      "coulomb_gradient: the screening threshold tau is nan");
}

} // namespace
