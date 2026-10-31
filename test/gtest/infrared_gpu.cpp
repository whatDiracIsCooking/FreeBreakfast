// GPU cases for hessian::rhf_dipole_derivative and hessian::infrared_intensities,
// and the [hessian] step that reports them:
//
// - the dipole derivative of the skew water, the skew formaldehyde and the
//   water at PySCF's DF-RHF minimum in cc-pVDZ / cc-pVDZ-JKFIT, built from
//   PySCF's orbitals and CPHF solutions (data/rhf_hessian/), against PySCF's
//   (data/dipole_derivative/, tools/emit/dipole_derivative_reference.py, itself
//   checked there against central differences of PySCF's SCF dipole), and its
//   sum over the atoms;
// - the analytic [hessian] at that minimum, run by Calculation from a TOML
//   config: its dipole derivative, frequencies and IR intensities against
//   PySCF's, and hessian.json holding the same;
// - the intensity of a hand-made mode, and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

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
import free_breakfast.integrals;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

// After the imports: it names std::filesystem from this unit's `import std;`.
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The dipole derivative (e) against PySCF's from the same C and U, absolute;
// max |dmu/dx| is 0.7 to 1.1. The issue's bar is 1e-6.
constexpr double kDipoleDerivativeTol = 1e-8;
// Sum_A dmu_c/dR_{A,k} of a neutral molecule: zero to the CPHF's tolerance.
constexpr double kSumRuleTol = 1e-8;
// The step's own SCF and CPHF against PySCF's: the SCF's convergence
// (energy 1e-12, commutator 1e-10) bounds the dipole derivative.
constexpr double kStepDipoleDerivativeTol = 1e-8;
// The issue's bar for the intensities (km/mol), and the frequencies (cm^-1)
// as pipeline_hessian_gpu.cpp holds them.
constexpr double kIntensityTol = 0.1;
constexpr double kFrequencyTol = 0.1;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR};

fb::json find_case(const std::filesystem::path &file, const char *list, const std::string &name) {
  const auto doc = fb::read_json(kDataDir / file);
  EXPECT_TRUE(doc) << doc.error().message;
  if (doc) {
    for (const fb::json &c : (*doc)[list]) {
      if (c["name"].get<std::string>() == name) {
        return c;
      }
    }
  }
  ADD_FAILURE() << "no case " << name << " in " << file;
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

std::vector<double> flat_rows(const fb::json &rows) {
  std::vector<double> flat;
  for (const auto &row : rows) {
    for (const double v : row.get<std::vector<double>>()) {
      flat.push_back(v);
    }
  }
  return flat;
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// max over k and c of |Sum_A dmu_c/dR_{A,k}|, of a 3 n_atom x 3 row-major.
double sum_rule(const std::vector<double> &d) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t c = 0; c < 3; ++c) {
      double sum = 0.0;
      for (std::size_t x = k; 3 * x < d.size(); x += 3) {
        sum += d[(3 * x) + c];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_rhf_hessian_gpu.cpp).
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

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(std::max<std::size_t>(host.size(), 1), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// One case's molecule, basis and classes, and PySCF's C (rows over N, made
// ours) and U, from data/rhf_hessian/.
class HessianDipoleDerivative : public ::testing::Test {
protected:
  void load(const std::string &name) {
    hessian_case_ = find_case("rhf_hessian/rhf_hessian.json", "cases", name);
    want_ = find_case("dipole_derivative/dipole_derivative.json", "rhf", name);
    ASSERT_FALSE(hessian_case_.is_null() || want_.is_null());
    res_.emplace(0);
    molecule_.emplace(*res_, case_atoms(hessian_case_));
    const auto stamp = fb::resolve_basis(hessian_case_["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    nao_ = basis_->nao_sphe();
    ASSERT_EQ(nao_, hessian_case_["nao"].get<std::size_t>());
    nocc_ = hessian_case_["nocc"].get<std::int32_t>();

    const auto c_file =
        fb::read_npy(kDataDir / "rhf_hessian" / hessian_case_["c"].get<std::string>());
    ASSERT_TRUE(c_file);
    std::vector<double> c = c_file->data;
    ASSERT_EQ(c.size(), nao_ * nao_);
    std::vector<double> norms;
    for (const fb::Shell &shell : basis_->shells()) {
      norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
    }
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t m = 0; m < nao_; ++m) {
        c[m + (nao_ * j)] /= norms[m];
      }
    }
    c_.emplace(upload(*res_, c));
    const auto u_file =
        fb::read_npy(kDataDir / "rhf_hessian" / hessian_case_["u"].get<std::string>());
    ASSERT_TRUE(u_file);
    u_.emplace(upload(*res_, u_file->data));
  }

  fb::json hessian_case_;
  fb::json want_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::Buffers::device<double>> c_;
  std::optional<fb::Buffers::device<double>> u_;
  std::size_t nao_ = 0;
  std::int32_t nocc_ = 0;
};

class HessianDipoleDerivativeGolden : public HessianDipoleDerivative,
                                      public ::testing::WithParamInterface<const char *> {};

TEST_P(HessianDipoleDerivativeGolden, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  load(name);
  ASSERT_FALSE(HasFatalFailure());
  const auto got =
      fb::hessian::rhf_dipole_derivative(*res_, *molecule_, *basis_, classes_, *c_, nocc_, *u_);
  ASSERT_TRUE(got) << got.error().message;
  const std::vector<double> want = flat_rows(want_["dipole_derivative"]);
  ASSERT_EQ(got->size(), want.size());
  const double worst = max_diff(*got, want);
  EXPECT_LE(worst, kDipoleDerivativeTol);
  const double drift = sum_rule(*got);
  EXPECT_LE(drift, kSumRuleTol);
  std::cout << std::format("[ dipole_d ] {}: |dmu/dx - PySCF| {:.2e}, sum rule {:.2e}\n", name,
                           worst, drift);
}

INSTANTIATE_TEST_SUITE_P(DfRhf, HessianDipoleDerivativeGolden,
                         ::testing::Values("h2o_ccpvdz", "h2co_ccpvdz", "h2o_ccpvdz_opt"));

TEST_F(HessianDipoleDerivative, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_ccpvdz");
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<std::vector<double>> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::rhf_dipole_derivative:"), std::string::npos) << what;
  };
  expect_refused(
      fb::hessian::rhf_dipole_derivative(*res_, *molecule_, *basis_, classes_, *c_, 0, *u_),
      "nocc 0");
  const fb::Buffers::device<double> short_u(u_->num_elements() - 1, res_->device());
  expect_refused(
      fb::hessian::rhf_dipole_derivative(*res_, *molecule_, *basis_, classes_, *c_, nocc_, short_u),
      "U too short");
  const fb::Buffers::device<double> ragged_c(c_->num_elements() - 1, res_->device());
  expect_refused(fb::hessian::rhf_dipole_derivative(*res_, *molecule_, *basis_, classes_, ragged_c,
                                                    nocc_, *u_),
                 "C not whole columns");
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  expect_refused(fb::hessian::rhf_dipole_derivative(*res_, two, *basis_, classes_, *c_, nocc_, *u_),
                 "another molecule");

  // An f shell: the dipole integrals stop at d.
  const auto stamp = fb::resolve_basis("cc-pvtz");
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto tz = fb::place(stamp->primitive_normalized(), *molecule_);
  ASSERT_TRUE(tz) << tz.error().message;
  const auto tz_classes = fb::primitive_pair_classes(*res_, *tz);
  ASSERT_TRUE(tz_classes) << tz_classes.error().message;
  const std::size_t n = tz->nao_sphe();
  const auto occ = static_cast<std::size_t>(nocc_);
  const fb::Buffers::device<double> tz_c(n * n, res_->device());
  const fb::Buffers::device<double> tz_u((n - occ) * occ * 9, res_->device());
  const auto f =
      fb::hessian::rhf_dipole_derivative(*res_, *molecule_, *tz, *tz_classes, tz_c, nocc_, tz_u);
  ASSERT_FALSE(f);
  EXPECT_EQ(f.error().kind, fb::ErrorKind::InvalidConfig) << f.error().message;
}

// I = K |Sum_x L_x dmu/dx|^2 for one hand-made mode over two atoms, and a
// mode's sign not mattering; the refusals of mismatched shapes.
TEST(HessianInfraredIntensities, ProjectsOntoTheModes) {
  // dmu/dx = row x: (x + 1, 0, 2 x), x over 6 coordinates.
  std::vector<double> derivative(18);
  for (std::size_t x = 0; x < 6; ++x) {
    derivative[3 * x] = static_cast<double>(x + 1);
    derivative[(3 * x) + 2] = 2.0 * static_cast<double>(x);
  }
  const std::vector<double> modes{0.5, 0.0, 0.0, 0.0, 0.0, -0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.25};
  const auto got = fb::hessian::infrared_intensities(modes, derivative);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->size(), 2U);
  // dmu/dQ = 0.5 (1, 0, 0) - 0.25 (6, 0, 10) = (-1, 0, -2.5).
  constexpr double kKmPerMol = 974.8801111743352;
  EXPECT_NEAR((*got)[0], kKmPerMol * 7.25, 1e-10);
  EXPECT_EQ((*got)[0], (*got)[1]);

  EXPECT_FALSE(
      fb::hessian::infrared_intensities(modes, std::span<const double>{derivative}.first(17)));
  EXPECT_FALSE(
      fb::hessian::infrared_intensities(std::span<const double>{modes}.first(11), derivative));
  EXPECT_FALSE(fb::hessian::infrared_intensities(modes, {}));
}

// The analytic [hessian] at PySCF's DF-RHF minimum of water: its dipole
// derivative, frequencies and intensities against PySCF's, and hessian.json.
TEST(PipelineInfrared, AnalyticHessianReportsIntensities) {
  FBF_REQUIRE_GPU();
  const fb::json want =
      find_case("dipole_derivative/dipole_derivative.json", "rhf", "h2o_ccpvdz_opt");
  ASSERT_FALSE(want.is_null());
  const fb::test::ScratchDir dir("pipeline_infrared");
  std::string xyz = std::format("{}\nh2o_ccpvdz_opt\n", want["atoms"].size());
  for (const fb::json &atom : want["atoms"]) {
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
                  "[hessian]\n",
                  want["basis"].get<std::string>(), want["aux_basis"].get<std::string>());
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const auto start = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const auto &out = calc->get<fb::HessianOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;

  const double d_err = max_diff(out->dipole_derivative, flat_rows(want["dipole_derivative"]));
  EXPECT_LE(d_err, kStepDipoleDerivativeTol);
  const auto frequencies = want["frequencies"].get<std::vector<double>>();
  const auto intensities = want["intensities"].get<std::vector<double>>();
  ASSERT_EQ(out->vibrations.frequencies.size(), frequencies.size());
  ASSERT_EQ(out->intensities.size(), intensities.size());
  const double nu_err = max_diff(out->vibrations.frequencies, frequencies);
  EXPECT_LE(nu_err, kFrequencyTol);
  const double i_err = max_diff(out->intensities, intensities);
  EXPECT_LE(i_err, kIntensityTol);

  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written) << written.error().message;
  EXPECT_EQ((*written)["intensities"].get<std::vector<double>>(), out->intensities);
  EXPECT_EQ(flat_rows((*written)["dipole_derivative"]), out->dipole_derivative);
  std::cout << std::format("[ infrared ] h2o_ccpvdz_opt: intensities {:.4f} {:.4f} {:.4f} km/mol, "
                           "|I - PySCF| {:.2e}, |dmu/dx - PySCF| {:.2e}, |nu - PySCF| {:.2e}; "
                           "{:.2f} s\n",
                           out->intensities[0], out->intensities[1], out->intensities[2], i_err,
                           d_err, nu_err, elapsed);
}

} // namespace
