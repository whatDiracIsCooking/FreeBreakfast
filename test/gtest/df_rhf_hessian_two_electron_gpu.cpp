// GPU cases for hessian::df_rhf_two_electron, the two-electron explicit part of
// the DF-RHF nuclear Hessian: its Coulomb and exchange terms against
// pyscf.df.hessian.rhf's _partial_hess_ejk ej and -ek over the same fitting
// set (data/rhf_hessian/, tools/emit/rhf_hessian_reference.py), for the skew
// water and a skew formaldehyde in cc-pVDZ / cc-pVDZ-JKFIT; their symmetry and
// the translational sum rule; and the refusals.

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
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar against PySCF, absolute.
constexpr double kHessianTol = 1e-9;
// Each term's asymmetry and its sum over atoms, absolute: the cross terms are
// symmetric by construction and the second-derivative integrals to rounding;
// the sum rule is the integrals' (held to 1e-10 of max |H| per class in
// df_hessian_gpu.cpp).
constexpr double kSymmetryTol = 1e-12;
constexpr double kSumRuleTol = 1e-9;

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

std::vector<double> read_array(const std::string &file, const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
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

// max |A - A^T| of an n x n matrix.
double asymmetry(const std::vector<double> &a, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t y = 0; y < n; ++y) {
    for (std::size_t x = 0; x < n; ++x) {
      worst = std::max(worst, std::abs(a[x + (n * y)] - a[y + (n * x)]));
    }
  }
  return worst;
}

// max over k and y of |Sum_A H[3 A + k][y]|: translational invariance.
double sum_rule(const std::vector<double> &h, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t y = 0; y < n; ++y) {
      double sum = 0.0;
      for (std::size_t x = k; x < n; x += 3) {
        sum += h[(x * n) + y];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

fb::Molecule molecule_of(fb::DeviceResources &res, const fb::json &c) {
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  return fb::Molecule(res, std::move(atoms));
}

// One case set up as the DF steps would: the bases placed, the fit B and its
// L, PySCF's orbitals made ours (rows over N), and the reference.
class HessianDfRhfTwoElectron : public ::testing::Test {
protected:
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "rhf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    const fb::json *c = nullptr;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        c = &entry;
      }
    }
    ASSERT_NE(c, nullptr) << "no case " << name;
    res_.emplace(0);
    molecule_.emplace(molecule_of(*res_, *c));
    const auto stamp = fb::resolve_basis((*c)["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    aux_stamp_ = (*c)["aux_basis"].get<std::string>();
    const auto aux_stamp = fb::resolve_aux_basis(aux_stamp_);
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(aux) << aux.error().message;
    aux_.emplace(std::move(*aux));
    nao_ = (*c)["nao"].get<std::size_t>();
    naux_ = (*c)["naux"].get<std::size_t>();
    nocc_ = (*c)["nocc"].get<std::int32_t>();
    ASSERT_EQ(basis_->nao_sphe(), nao_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    n_x_ = 3 * basis_->natoms();

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
    std::vector<double> c_host = read_array((*c)["c"].get<std::string>(), {nao_, nao_});
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t m = 0; m < nao_; ++m) {
        c_host[m + (nao_ * j)] /= n_ao[m];
      }
    }
    c_.emplace(upload(*res_, c_host));
    ref_coulomb_ = read_array((*c)["coulomb"].get<std::string>(), {n_x_, n_x_});
    ref_exchange_ = read_array((*c)["exchange"].get<std::string>(), {n_x_, n_x_});
  }

  fb::Result<fb::hessian::RhfTwoElectron> run(const fb::BasisSet &aux, const std::int32_t nocc,
                                              const fb::Buffers::device<double> &b,
                                              const fb::Buffers::device<double> &c) {
    return fb::hessian::df_rhf_two_electron(*res_, *basis_, classes_, aux, b,
                                            metric_->spherical().data(), c, nocc);
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    const auto start = std::chrono::steady_clock::now();
    auto got = run(*aux_, nocc_, *b_, *c_);
    ASSERT_TRUE(got) << got.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_EQ(got->coulomb.size(), n_x_ * n_x_);
    ASSERT_EQ(got->exchange.size(), n_x_ * n_x_);
    // The call's own buffers are freed on return; what stays is the
    // workspace, grown to the largest scratch any of its calls took.
    const std::size_t workspace = res_->device_workspace(0).num_elements();

    std::string report;
    for (const auto &[term, mine, ref] : {std::tuple{"coulomb", &got->coulomb, &ref_coulomb_},
                                          std::tuple{"exchange", &got->exchange, &ref_exchange_}}) {
      const double scale = max_abs(*ref);
      const double worst = max_diff(*mine, *ref);
      const double asym = asymmetry(*mine, n_x_);
      const double drift = sum_rule(*mine, n_x_);
      EXPECT_LE(worst, kHessianTol) << term << ": max |H - PySCF| " << worst << " of " << scale;
      EXPECT_LE(asym, kSymmetryTol) << term;
      EXPECT_LE(drift, kSumRuleTol) << term;
      report += std::format("; {} max |H| {:.4f}, |H - PySCF| {:.2e}, asym {:.2e}, sum rule "
                            "{:.2e}",
                            term, scale, worst, asym, drift);
    }
    std::cout << std::format("[ hess2e   ] {} ({}): {} coordinates, {:.2f} s, workspace {} "
                             "bytes{}\n",
                             name, aux_stamp_, n_x_, elapsed, workspace, report);
  }

  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
  std::vector<double> ref_coulomb_;
  std::vector<double> ref_exchange_;
  std::string aux_stamp_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t nocc_ = 0;
};

TEST_F(HessianDfRhfTwoElectron, WaterMatchesPyscf) {
  check("h2o_ccpvdz");
}

TEST_F(HessianDfRhfTwoElectron, FormaldehydeMatchesPyscf) {
  check("h2co_ccpvdz");
}

TEST_F(HessianDfRhfTwoElectron, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_ccpvdz");
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::RhfTwoElectron> &r,
                                 const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_rhf_two_electron"), std::string::npos) << what;
  };
  expect_refused(run(*aux_, 0, *b_, *c_), "nocc 0");
  expect_refused(run(*aux_, static_cast<std::int32_t>(nao_) + 1, *b_, *c_), "nocc past nao");
  const fb::Buffers::device<double> short_c(nao_ * nocc_ - 1, res_->device());
  expect_refused(run(*aux_, nocc_, *b_, short_c), "C shorter than nao nocc");
  const fb::Buffers::device<double> short_b(b_->num_elements() - 1, res_->device());
  expect_refused(run(*aux_, nocc_, short_b, *c_), "B not naux nao^2");
  // The fitting set on two of the three atoms.
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  const auto aux_stamp = fb::resolve_aux_basis(aux_stamp_);
  ASSERT_TRUE(aux_stamp);
  const auto aux_two = fb::place(aux_stamp->primitive_normalized(), two);
  ASSERT_TRUE(aux_two);
  expect_refused(run(*aux_two, nocc_, *b_, *c_), "aux on other atoms");
}

} // namespace
