// GPU cases for hessian::df_rhf, the full DF-RHF analytic nuclear Hessian:
// the total and each of its terms against pyscf.df.hessian.rhf over the same
// fitting set (data/rhf_hessian/, tools/emit/rhf_hessian_reference.py), for
// the skew water and a skew formaldehyde in cc-pVDZ / cc-pVDZ-JKFIT and the
// skew water in cc-pVTZ / cc-pVTZ-JKFIT (f shells); its symmetry and the
// translational sum rule; against a central difference of the tree's own
// gradient::df_rhf (the [hessian] step, run here on the same water); at
// PySCF's minimum, six near-zero mass-weighted eigenvalues and the harmonic
// frequencies; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.geometry;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

// After the imports: it names std::filesystem from this unit's `import std;`.
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar against PySCF, absolute, for the total and each term (max
// |H| is 0.7 to 1.0). The parts measured on their own are 1e-11 to 1e-10 from
// PySCF; the response part's CPHF, stopped near 1e-10 in the residual, is the
// loosest.
constexpr double kHessianTol = 1e-7;
// The terms' sum before symmetrizing, of max |H|: each term is symmetric to
// rounding, the one-electron off-diagonal blocks (A, B) and (B, A) being
// accumulated separately by atomics.
constexpr double kAsymmetryRelTol = 1e-10;
// Sum_A H[3 A + k][y], absolute: the response part's sum over its second index
// holds only to the CPHF tolerance (rhf_hessian.tex, "Checks").
constexpr double kSumRuleTol = 1e-8;
constexpr double kCphfTolerance = 1e-10;
// Against central differences of the gradient at h = 1e-3 bohr: their h^2
// error and the SCF's noise over 2h (pipeline_hessian_gpu.cpp's bar).
constexpr double kCentralDifferenceTol = 1e-5;
// At the minimum, the six rigid-body eigenvalues of the mass-weighted Hessian
// (hartree / bohr^2 / amu), against a bend's 0.1: the Hessian's own error over
// the lightest mass, the gradient there being 1.5e-12.
constexpr double kRigidEigenvalueTol = 1e-6;
// The harmonic frequencies against PySCF's of its own Hessian (cm^-1), as
// geometry_frequencies_gpu.cpp holds harmonic_analysis.
constexpr double kFrequencyTol = 1e-3;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "rhf_hessian";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_jk_derivative_gpu.cpp).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : shell.primitives()) {
    for (const fb::Primitive &q : shell.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

// Each spherical function's norm N_i: function i here is N_i times PySCF's
// unit-normalised one (test/gtest/data/README.md, "Normalisation").
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> download(fb::DeviceResources &res, const double *device,
                             const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

std::vector<double> read_array(const std::string &file, const std::size_t n) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, (std::vector<std::size_t>{n, n})) << file;
  return a->data;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// max over k and y of |Sum_A H[3 A + k][y]| of an n x n matrix.
double sum_rule(const std::vector<double> &h, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t y = 0; y < n; ++y) {
    for (std::size_t k = 0; k < 3; ++k) {
      double sum = 0.0;
      for (std::size_t a = 0; a < n / 3; ++a) {
        sum += h[(3 * a) + k + (n * y)];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

// One case set up as the DF steps would: the bases placed, the fit B and its
// L, PySCF's orbitals made ours (rows over N), and the reference.
class HessianDfRhf : public ::testing::Test {
protected:
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "rhf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        case_ = entry;
      }
    }
    ASSERT_FALSE(case_.is_null()) << "no case " << name;
    res_.emplace(0);
    for (const auto &atom : case_["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms_.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                                static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
    }
    molecule_.emplace(*res_, atoms_);
    const auto stamp = fb::resolve_basis(case_["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    const auto aux_stamp = fb::resolve_aux_basis(case_["aux_basis"].get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(aux) << aux.error().message;
    aux_.emplace(std::move(*aux));
    nao_ = case_["nao"].get<std::size_t>();
    naux_ = case_["naux"].get<std::size_t>();
    nocc_ = case_["nocc"].get<std::int32_t>();
    ASSERT_EQ(basis_->nao_sphe(), nao_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    n_x_ = 3 * basis_->natoms();
    energies_ = case_["mo_energies"].get<std::vector<double>>();

    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto u = fb::upload_spherical_transform_matrix(*res_, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(*res_, *aux_, std::move(*u));
    b_.emplace(naux_ * nao_ * nao_, res_->device());
    ASSERT_TRUE(fb::metric(*res_, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(*res_, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(*res_, static_cast<std::int32_t>(nao_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));

    const std::vector<double> n_ao = norms(*basis_);
    const auto c_file = fb::read_npy(kDataDir / case_["c"].get<std::string>());
    ASSERT_TRUE(c_file);
    std::vector<double> c = c_file->data;
    ASSERT_EQ(c.size(), nao_ * nao_);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t m = 0; m < nao_; ++m) {
        c[m + (nao_ * j)] /= n_ao[m];
      }
    }
    c_.emplace(upload(*res_, c));
  }

  fb::Result<fb::hessian::RhfTerms> run() {
    return fb::hessian::df_rhf(*res_, *molecule_, *basis_, classes_, *aux_, *b_,
                               metric_->spherical().data(), *c_, energies_, nocc_, kCphfTolerance);
  }

  std::vector<double> reference(const char *term) {
    return read_array(case_[term].get<std::string>(), n_x_);
  }

  // The total and every term against PySCF, the total's symmetry and sum
  // rule, and the CPHF's residuals; the Hessian.
  std::vector<double> check(const std::string &name) {
    load(name);
    if (HasFatalFailure()) {
      return {};
    }
    const auto start = std::chrono::steady_clock::now();
    auto got = run();
    EXPECT_TRUE(got) << got.error().message;
    if (!got) {
      return {};
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_EQ(got->hessian.size(), n_x_ * n_x_);
    EXPECT_EQ(got->response.cphf.residuals.size(), n_x_);
    for (const double r : got->response.cphf.residuals) {
      EXPECT_LE(r, 10 * kCphfTolerance);
    }
    const std::vector<double> want = reference("total");
    const double scale = max_abs(want);
    const double worst = max_diff(got->hessian, want);
    EXPECT_LE(worst, kHessianTol) << "max |H - PySCF| " << worst << " of max " << scale;
    std::string terms;
    const std::array<std::pair<const char *, const std::vector<double> *>, 5> parts{{
        {"one_electron", &got->one_electron},
        {"nuclear_repulsion", &got->nuclear_repulsion},
        {"coulomb", &got->two_electron.coulomb},
        {"exchange", &got->two_electron.exchange},
        {"response", &got->response.hessian},
    }};
    for (const auto &[term, value] : parts) {
      const double err = max_diff(*value, reference(term));
      EXPECT_LE(err, kHessianTol) << term;
      terms += std::format(" {} {:.1e}", term, err);
    }
    for (std::size_t x = 0; x < n_x_; ++x) {
      for (std::size_t y = 0; y < n_x_; ++y) {
        EXPECT_EQ(got->hessian[x + (n_x_ * y)], got->hessian[y + (n_x_ * x)]);
      }
    }
    EXPECT_LE(got->asymmetry, kAsymmetryRelTol * scale);
    const double drift = sum_rule(got->hessian, n_x_);
    EXPECT_LE(drift, kSumRuleTol);
    std::cout << std::format(
        "[ hess     ] {}: {} coordinates, {} sweeps, {:.2f} s; max |H| {:.4f}, "
        "|H - PySCF| {:.2e} ({}), asym {:.2e}, sum rule {:.2e}\n",
        name, n_x_, got->response.cphf.iterations, elapsed, scale, worst, terms.substr(1),
        got->asymmetry, drift);
    return std::move(got->hessian);
  }

  fb::json case_;
  std::vector<fb::Atom> atoms_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
  std::vector<double> energies_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t nocc_ = 0;
};

TEST_F(HessianDfRhf, WaterMatchesPyscf) {
  FBF_REQUIRE_GPU();
  check("h2o_ccpvdz");
}

TEST_F(HessianDfRhf, FormaldehydeMatchesPyscf) {
  FBF_REQUIRE_GPU();
  check("h2co_ccpvdz");
}

// The f shells. 11 s on an idle RTX 3080, so slow-gated: the cc-pVDZ water
// stands for the family in the fast tier.
TEST_F(HessianDfRhf, WaterTripleZetaMatchesPyscf) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  check("h2o_ccpvtz");
}

// The tree against itself: the [hessian] step's central differences of
// gradient::df_rhf, each displacement a full DF-RHF SCF of its own, on the
// same skew water and sets, configured as pipeline_hessian_gpu.cpp does.
TEST_F(HessianDfRhf, WaterMatchesCentralDifference) {
  FBF_REQUIRE_GPU();
  const std::vector<double> analytic = check("h2o_ccpvdz");
  ASSERT_FALSE(HasFailure());

  const fb::test::ScratchDir dir("df_rhf_hessian");
  std::string xyz = std::format("{}\nh2o_ccpvdz\n", atoms_.size());
  for (const fb::json &atom : case_["atoms"]) {
    const auto coord = atom["coord"].get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom["symbol"].get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
                  "[hessian]\nmethod = \"numerical\"\n",
                  case_["basis"].get<std::string>(), case_["aux_basis"].get<std::string>());
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &numerical = calc->get<fb::HessianOutput>();
  ASSERT_TRUE(numerical.has_value()) << numerical.error().message;
  ASSERT_EQ(numerical->hessian.size(), analytic.size());
  const double worst = max_diff(numerical->hessian, analytic);
  EXPECT_LE(worst, kCentralDifferenceTol);
  std::cout << std::format("[ hess     ] h2o_ccpvdz analytic vs central differences of "
                           "gradient::df_rhf: max {:.2e}\n",
                           worst);
}

// At PySCF's minimum (no gradient component above 1.5e-12), the mass-weighted
// Hessian has six near-zero eigenvalues, the three translations and three
// rotations, and harmonic_analysis gives PySCF's frequencies.
TEST_F(HessianDfRhf, MinimumHasSixRigidModes) {
  FBF_REQUIRE_GPU();
  const std::vector<double> h = check("h2o_ccpvdz_opt");
  ASSERT_FALSE(HasFailure());
  const auto masses = case_["masses"].get<std::vector<double>>();
  ASSERT_EQ(masses.size(), atoms_.size());
  std::vector<double> weighted(h.size());
  for (std::size_t x = 0; x < n_x_; ++x) {
    for (std::size_t y = 0; y < n_x_; ++y) {
      weighted[x + (n_x_ * y)] = h[x + (n_x_ * y)] / std::sqrt(masses[x / 3] * masses[y / 3]);
    }
  }
  fb::Buffers::device<double> a = upload(*res_, weighted);
  fb::Buffers::device<double> w(n_x_, res_->device());
  ASSERT_TRUE(fb::diagonalize(*res_, static_cast<std::int32_t>(n_x_), a.data(), w.data()));
  std::vector<double> eigenvalues = download(*res_, w.data(), n_x_);
  std::ranges::sort(eigenvalues, {}, [](const double v) { return std::abs(v); });
  constexpr std::size_t kRigid = 6;
  const double rigid = std::abs(eigenvalues[kRigid - 1]);
  EXPECT_LE(rigid, kRigidEigenvalueTol);
  // The smallest vibration, the bend, is 0.1: far from the rigid six.
  EXPECT_GT(eigenvalues[kRigid], 1e4 * kRigidEigenvalueTol);

  const auto vib = fb::geometry::harmonic_analysis(*res_, atoms_, h);
  ASSERT_TRUE(vib) << vib.error().message;
  const auto want = case_["frequencies"].get<std::vector<double>>();
  ASSERT_EQ(vib->frequencies.size(), want.size());
  double worst = 0.0;
  for (std::size_t j = 0; j < want.size(); ++j) {
    EXPECT_NEAR(vib->frequencies[j], want[j], kFrequencyTol) << "mode " << j;
    worst = std::max(worst, std::abs(vib->frequencies[j] - want[j]));
  }
  std::cout << std::format("[ hess     ] h2o_ccpvdz_opt: largest rigid eigenvalue {:.2e}, "
                           "smallest vibration {:.4f}; frequencies {:.2f} {:.2f} {:.2f} cm^-1, "
                           "|nu - PySCF| {:.2e}\n",
                           rigid, eigenvalues[kRigid], vib->frequencies[0], vib->frequencies[1],
                           vib->frequencies[2], worst);
}

TEST_F(HessianDfRhf, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_ccpvdz");
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::RhfTerms> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_rhf:"), std::string::npos) << what;
  };
  const std::span<const double> energies{energies_};
  const double *l = metric_->spherical().data();
  expect_refused(
      fb::hessian::df_rhf(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l, *c_, energies, 0),
      "nocc 0");
  expect_refused(fb::hessian::df_rhf(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l, *c_,
                                     energies, static_cast<std::int32_t>(nao_)),
                 "no virtuals");
  expect_refused(fb::hessian::df_rhf(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l, *c_,
                                     energies.first(energies.size() - 1), nocc_),
                 "C not nao nmo");
  const fb::Buffers::device<double> short_b(b_->num_elements() - 1, res_->device());
  expect_refused(fb::hessian::df_rhf(*res_, *molecule_, *basis_, classes_, *aux_, short_b, l, *c_,
                                     energies, nocc_),
                 "B not naux nao^2");
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  expect_refused(
      fb::hessian::df_rhf(*res_, two, *basis_, classes_, *aux_, *b_, l, *c_, energies, nocc_),
      "another molecule");
}

} // namespace
