// GPU cases for hessian::df_casscf, the state-specific DF-CASSCF analytic
// nuclear Hessian (casscf_hessian.tex eq. ss), against data/casscf_hessian/
// (tools/emit/casscf_hessian_reference.py, over the same cc-pVDZ-JKFIT fit) at
// the fixture's orbitals and root, loaded, never re-converged (PySCF's own
// cc-pVDZ solution is a saddle point): the stretched water CAS(4,4) in 6-31G
// and in cc-pVDZ (d shells). E_xy, the response term, lambda^x's orbital part
// and the total against the emitter's, and the total against its relaxed_fd,
// the analytic relaxed gradient differenced at re-solved displaced
// geometries; the CP residuals, the symmetry before symmetrizing and the
// translational sum rule. The fixture's geometry is not a stationary point
// (|dE/dx| up to 0.074), so the six near-zero mass-weighted eigenvalues of a
// minimum are not checked here. Beside them: an excited root (the second
// CASCI root at the fixture's orbitals, a saddle in the parameters) refused,
// and the input refusals.

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
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.mcscf.zvector;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// E_xy against the emitter's, absolute: df_casscf_exy's own test's bar.
constexpr double kExyTol = 1e-8;
// The response term and the total against the emitter's, absolute: the
// issue's 1e-7. lambda^x is good to the CI response's tolerance (1e-8 of its
// right-hand side), which bounds the response term.
constexpr double kHessianTol = 1e-7;
// lambda^x's orbital part, relative to its largest element.
constexpr double kLambdaTol = 1e-7;
// Against relaxed_fd (a fourth-order difference, h = 1e-3 bohr, of the
// relaxed gradient re-solved at each displacement): the design's 1e-6.
constexpr double kRelaxedFdTol = 1e-6;
// Each CP column's unreduced residual, df_casscf's own check of it.
constexpr double kResidualTol = 1e-8;
// max |H - H^T| before symmetrizing, and the total's sum over atoms.
constexpr double kAsymmetryTol = 1e-8;
constexpr double kSumRuleTol = 1e-7;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

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

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
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

// One case set up as the DF steps would: the bases placed, h, the fit B and
// its L, the fixture's orbitals made ours (rows over N, in this tree's AO
// order), and the CASSCF's integrals and sigma at them, for a root's energy.
class HessianDfCasscf : public ::testing::Test {
protected:
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "casscf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        case_ = entry;
      }
    }
    ASSERT_FALSE(case_.is_null()) << "no case " << name;
    res_.emplace(0);
    fb::DeviceResources &res = *res_;
    std::vector<fb::Atom> atoms;
    for (const auto &atom : case_["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                               static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
    }
    molecule_.emplace(res, std::move(atoms));
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
    ASSERT_EQ(basis_->nao_sphe(), nao_);
    ASSERT_EQ(case_["nmo"].get<std::size_t>(), nao_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    ASSERT_EQ(case_["weights"].size(), 1U);
    n_x_ = 3 * basis_->natoms();
    ncore_ = case_["ncore"].get<std::int32_t>();
    ncas_ = case_["ncas"].get<std::int32_t>();
    const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
    nalpha_ = nelecas[0];
    nbeta_ = nelecas[1];
    n_det_ = case_["na"].get<std::size_t>() * case_["nb"].get<std::size_t>();

    auto classes = fb::primitive_pair_classes(res, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto su = fb::upload_spherical_transform_matrix(res, *basis_);
    ASSERT_TRUE(su) << su.error().message;
    h_.emplace(res, *basis_, std::move(*su));
    ASSERT_TRUE(fb::core_hamiltonian(res, classes_, *molecule_, h_->cartesian()));
    ASSERT_TRUE(h_->cartesian_to_spherical(res));
    auto u = fb::upload_spherical_transform_matrix(res, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(res, *aux_, std::move(*u));
    b_.emplace(naux_ * nao_ * nao_, res.device());
    ASSERT_TRUE(fb::metric(res, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(res, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nao_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));

    // PySCF orders each atom's shells by l, keeping the file's order within
    // one l; this tree keeps the file's order, so a split SP shell (6-31G's)
    // puts 2p before 3s here and after it there. Row k of the fixture's C is
    // this tree's function order[k].
    std::vector<double> norms;
    for (const fb::Shell &shell : basis_->shells()) {
      norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
    }
    std::vector<std::size_t> order;
    for (std::uint32_t atom = 0; atom < basis_->natoms(); ++atom) {
      std::vector<std::uint32_t> shells(basis_->atom_shell_count(atom));
      std::iota(shells.begin(), shells.end(), basis_->atom_shell_offset(atom));
      std::ranges::stable_sort(shells, {}, [&](const std::uint32_t i) {
        return basis_->shells()[i].angular_momentum();
      });
      for (const std::uint32_t i : shells) {
        const std::size_t first = basis_->ao_offset_sphe(i);
        for (std::size_t f = 0; f < basis_->shells()[i].function_count_sphe(); ++f) {
          order.push_back(first + f);
        }
      }
    }
    ASSERT_EQ(order.size(), nao_);
    const std::vector<double> c_pyscf =
        read_array(case_["arrays"]["c"].get<std::string>(), {nao_, nao_});
    c_host_.assign(nao_ * nao_, 0.0);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t k = 0; k < nao_; ++k) {
        c_host_[order[k] + (nao_ * j)] = c_pyscf[k + (nao_ * j)] / norms[order[k]];
      }
    }
    ci_host_ = read_array(case_["arrays"]["ci"].get<std::string>(), {n_det_, 1});
    ASSERT_FALSE(HasFailure());

    // The active-space Hamiltonian at C, on the sigma (row-major host arrays
    // need no reorder: h_eff is symmetric and (tu|vw) column-major is (wv|ut)
    // row-major), for a root's energy and the excited root's CASCI.
    fb::SphericalMatrixAO c(res, *basis_);
    upload(res, c_host_, c.data());
    auto ints = fb::casscf_integrals(res, *basis_, *molecule_, h_->spherical(), *b_,
                                     static_cast<std::int32_t>(naux_), c, ncore_, ncas_);
    ASSERT_TRUE(ints) << ints.error().message;
    auto alpha = fb::ci::DeviceStringSet::create(res, ncas_, nalpha_);
    ASSERT_TRUE(alpha) << alpha.error().message;
    auto beta = fb::ci::DeviceStringSet::create(res, ncas_, nbeta_);
    ASSERT_TRUE(beta) << beta.error().message;
    auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
    ASSERT_TRUE(sigma) << sigma.error().message;
    const auto n = static_cast<std::size_t>(ncas_);
    ASSERT_TRUE(sigma->set_integrals(res, ints->hamiltonian.e_core,
                                     download(res, ints->hamiltonian.h_eff.data(), n * n),
                                     download(res, ints->hamiltonian.eri.data(), n * n * n * n)));
    alpha_.emplace(std::move(*alpha));
    beta_.emplace(std::move(*beta));
    sigma_.emplace(std::move(*sigma));
  }

  // A one-root mcscf::Solution at the fixture's orbitals: the CI vector
  // @p ci, <c|H_0|c> its energy and its RDMs by ci::Rdm.
  fb::mcscf::Solution solution(const std::vector<double> &ci) {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    fb::mcscf::Solution s{
        .c = fb::SphericalMatrixAO(res, *basis_),
        .roots = {{}, {0.0}, fb::Buffers::device<double>(n_det_, res.device()), {}, 0},
        .energy = 0.0,
        .gradient_norm = 0.0,
        .gamma = {},
        .gamma2 = {},
        .root_gammas = {},
        .iterations = {}};
    upload(res, c_host_, s.c.data());
    upload(res, ci, s.roots.vectors.data());
    fb::Buffers::device<double> hc(n_det_, res.device());
    EXPECT_TRUE(sigma_->apply(res, 1, s.roots.vectors.data(), hc.data()).has_value());
    const std::vector<double> hc_host = download(res, hc.data(), n_det_);
    s.energy = std::inner_product(ci.begin(), ci.end(), hc_host.begin(), 0.0);
    s.roots.energies.push_back(s.energy);
    auto rdm = fb::ci::Rdm::create(res, *alpha_, *beta_);
    EXPECT_TRUE(rdm) << rdm.error().message;
    fb::Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
    const std::vector<double> one{1.0};
    EXPECT_TRUE(rdm->rdm12(res, one, s.roots.vectors.data(), rdms.data(), rdms.data() + (n * n))
                    .has_value());
    s.gamma = download(res, rdms.data(), n * n);
    s.gamma2 = download(res, rdms.data() + (n * n), n * n * n * n);
    s.root_gammas = s.gamma;
    return s;
  }

  fb::Result<fb::hessian::CasscfTerms> run(const fb::mcscf::Solution &s, const std::int32_t ncore,
                                           const std::int32_t ncas) {
    return fb::hessian::df_casscf(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(), *b_,
                                  metric_->spherical().data(), s, ncore, ncas);
  }

  [[nodiscard]] std::vector<double> root_array(const std::string &key,
                                               const std::vector<std::size_t> &shape) const {
    return read_array(case_["roots"][0]["arrays"][key].get<std::string>(), shape);
  }

  // The fixture's lambda^x orbital part (K, nmo x nmo x n), packed as
  // zvectors' z_o.
  std::vector<double> fixture_lambda(const std::size_t pairs) {
    fb::DeviceResources &res = *res_;
    const std::vector<double> full =
        read_array(case_["arrays"]["lambda_orb"].get<std::string>(), {nao_, nao_, n_x_});
    fb::Buffers::device<double> device(full.size(), res.device());
    upload(res, full, device.data());
    fb::Buffers::device<double> packed(pairs * n_x_, res.device());
    for (std::size_t x = 0; x < n_x_; ++x) {
      EXPECT_TRUE(fb::response::pack_orbital_pairs(res, static_cast<std::int32_t>(nao_), ncore_,
                                                   ncas_, device.data() + (nao_ * nao_ * x),
                                                   packed.data() + (pairs * x))
                      .has_value());
    }
    return download(res, packed.data(), pairs * n_x_);
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_FALSE(HasFailure());
    const fb::mcscf::Solution s = solution(ci_host_);
    ASSERT_FALSE(HasFailure());
    EXPECT_NEAR(s.energy, case_["energies"][0].get<double>(), 1e-9);
    const auto start = std::chrono::steady_clock::now();
    auto got = run(s, ncore_, ncas_);
    ASSERT_TRUE(got) << got.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::vector<std::size_t> square{n_x_, n_x_};

    std::string report;
    for (const auto &[key, mine, tol] : {std::tuple{"exy", &got->exy.exy, kExyTol},
                                         std::tuple{"response", &got->response, kHessianTol},
                                         std::tuple{"hessian", &got->hessian, kHessianTol},
                                         std::tuple{"relaxed_fd", &got->hessian, kRelaxedFdTol}}) {
      const std::vector<double> ref = root_array(key, square);
      ASSERT_EQ(mine->size(), n_x_ * n_x_) << key;
      const double worst = max_diff(*mine, ref);
      EXPECT_LE(worst, tol) << key << ": max |H - ref| " << worst << " of " << max_abs(ref);
      report += std::format("; {} |H - ref| {:.2e} of {:.3e}", key, worst, max_abs(ref));
    }

    // lambda^x's orbital part against the emitter's.
    const std::size_t pairs = got->cp.orbital.num_elements() / n_x_;
    const std::vector<double> want = fixture_lambda(pairs);
    const std::vector<double> lambda = download(*res_, got->cp.orbital.data(), pairs * n_x_);
    const double lambda_miss = max_diff(lambda, want);
    EXPECT_LE(lambda_miss, kLambdaTol * std::max(1.0, max_abs(want)));

    ASSERT_EQ(got->residuals.size(), n_x_);
    const double residual = std::ranges::max(got->residuals);
    EXPECT_LE(residual, kResidualTol);
    EXPECT_LE(got->asymmetry, kAsymmetryTol);
    const double drift = sum_rule(got->hessian, n_x_);
    EXPECT_LE(drift, kSumRuleTol);
    std::cout << std::format("[ ss       ] {}: {} coordinates, {:.2f} s, CP residual {:.2e} "
                             "(iterations {}), lambda_orb |l - ref| {:.2e} of {:.3e}, asymmetry "
                             "{:.2e}, sum rule {:.2e}{}\n",
                             name, n_x_, elapsed, residual, std::ranges::max(got->cp.iterations),
                             lambda_miss, max_abs(want), got->asymmetry, drift, report);
  }

  fb::json case_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> h_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::ci::DeviceStringSet> alpha_;
  std::optional<fb::ci::DeviceStringSet> beta_;
  std::optional<fb::ci::Sigma> sigma_;
  std::vector<double> c_host_;
  std::vector<double> ci_host_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfCasscf, Water631gMatchesReference) {
  check("h2o_631g");
}

TEST_F(HessianDfCasscf, WaterCcpvdzMatchesReference) {
  check("h2o_ccpvdz");
}

// A state-specific excited root is a saddle in the parameters: the second
// CASCI root at the fixture's orbitals has the first below it in its own
// spin, so E_lambda lambda's CI block 2 (H_0 - E_1) is indefinite. The CP
// solve does not converge there, and the call says so rather than return a
// number.
TEST_F(HessianDfCasscf, ExcitedRootIsRefused) {
  FBF_REQUIRE_GPU();
  load("h2o_631g");
  ASSERT_FALSE(HasFatalFailure());
  fb::DeviceResources &res = *res_;
  fb::Buffers::device<double> diagonal(n_det_, res.device());
  ASSERT_TRUE(sigma_->diagonal(res, diagonal.data()));
  auto casci = fb::ci::solve(res, *sigma_, *alpha_, *beta_, diagonal.data(), 2);
  ASSERT_TRUE(casci) << casci.error().message;
  const std::vector<double> excited = download(res, casci->vectors.data() + n_det_, n_det_);
  const fb::mcscf::Solution s = solution(excited);
  ASSERT_FALSE(HasFailure());
  EXPECT_NEAR(s.energy, casci->energies[1], 1e-9);
  auto got = run(s, ncore_, ncas_);
  ASSERT_FALSE(got) << "an excited root returned a Hessian";
  EXPECT_EQ(got.error().kind, fb::ErrorKind::Divergence) << got.error().message;
  std::cout << std::format("[ ss       ] excited root ({:.6f} over {:.6f}): {}\n", s.energy,
                           casci->energies[0], got.error().message);
}

TEST_F(HessianDfCasscf, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g");
  ASSERT_FALSE(HasFatalFailure());
  const fb::mcscf::Solution s = solution(ci_host_);
  ASSERT_FALSE(HasFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::CasscfTerms> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_casscf"), std::string::npos) << what;
  };
  const auto nao = static_cast<std::int32_t>(nao_);
  expect_refused(run(s, -1, ncas_), "ncore -1");
  expect_refused(run(s, ncore_, 0), "ncas 0");
  expect_refused(run(s, nao - ncas_ + 1, ncas_), "ncore + ncas past nao");

  // Two roots: a state average is not this call's.
  fb::mcscf::Solution two = solution(ci_host_);
  two.roots.energies.push_back(two.energy);
  expect_refused(run(two, ncore_, ncas_), "two roots");
  // RDMs of the wrong size, and a trace that is no electron count.
  fb::mcscf::Solution short_gamma = solution(ci_host_);
  short_gamma.gamma.pop_back();
  expect_refused(run(short_gamma, ncore_, ncas_), "gamma not ncas^2");
  fb::mcscf::Solution odd = solution(ci_host_);
  odd.gamma[0] += 0.5;
  expect_refused(run(odd, ncore_, ncas_), "trace(gamma) not an electron count");
  // A root vector of the wrong length.
  fb::mcscf::Solution ragged = solution(ci_host_);
  ragged.roots.vectors = fb::Buffers::device<double>(n_det_ - 1, res_->device());
  expect_refused(run(ragged, ncore_, ncas_), "root not n_det long");
}

} // namespace
