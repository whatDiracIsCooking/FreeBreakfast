// GPU cases for hessian::df_sacasscf, a state-averaged DF-CASSCF root's
// analytic nuclear Hessian (casscf_hessian.tex eq. sa), against
// test/gtest/data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py,
// over the same cc-pVDZ-JKFIT fit): the stretched water's SA-CAS(4,4) / 6-31G
// at two roots (0.5, 0.5) and (0.7, 0.3), three equal and (0.5, 0.3, 0.2),
// every root, the orbitals and roots loaded from the fixture, never
// re-converged (PySCF's two-root solutions are saddle points). z and lambda^x
// are solved here. Each term against the fixture's (exy, exy_z, lxl_lambda,
// root_hessian_lambda, third, rot), the total against its hessian and its
// relaxed_fd (the analytic relaxed gradient differenced at re-solved displaced
// geometries); z's and lambda^x's orbital parts, the CP residuals, the symmetry
// before symmetrizing and the translational sum rule. At equal weights the
// total without eq. rot misses relaxed_fd; and the refusals.

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
// Every term built from z or lambda^x, and the total, against the emitter's,
// absolute: z and lambda^x are good to the CI response's tolerance (1e-8 of
// its right-hand side), as in df_casscf's test.
constexpr double kHessianTol = 1e-7;
// z's and lambda^x's orbital parts, relative to their largest element.
constexpr double kSolveTol = 1e-7;
// Against relaxed_fd: the design's 1e-6 (the issue asks ~1e-5).
constexpr double kRelaxedFdTol = 1e-6;
// Each CP column's unreduced residual, df_sacasscf's own check of it.
constexpr double kResidualTol = 1e-8;
// max |H - H^T| before symmetrizing, and the total's sum over atoms.
constexpr double kAsymmetryTol = 1e-8;
constexpr double kSumRuleTol = 1e-7;
// What the total misses relaxed_fd by without eq. rot at equal weights, at
// least: #609 measured 4.9e-2 (two roots) and 5.2e-2 to 4.8e-1 (three).
constexpr double kWithoutRotationMiss = 1e-2;

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

// One state-averaged case set up as the DF steps would: the bases placed, h,
// the fit B and its L, the fixture's orbitals made ours (rows over N, in this
// tree's AO order), and its roots as an mcscf::Solution at them (each
// <c_r|H_0|c_r> its energy, the averaged RDMs by ci::Rdm).
class HessianDfSaCasscf : public ::testing::Test {
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
    n_x_ = 3 * basis_->natoms();
    ncore_ = case_["ncore"].get<std::int32_t>();
    ncas_ = case_["ncas"].get<std::int32_t>();
    const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
    nalpha_ = nelecas[0];
    nbeta_ = nelecas[1];
    weights_ = case_["weights"].get<std::vector<double>>();
    nroots_ = weights_.size();
    ASSERT_GE(nroots_, 2U);
    equal_ = case_["equal_weights"].get<bool>();
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
    // one l; this tree keeps the file's order, so 6-31G's split SP shell puts
    // 2p before 3s here and after it there. Row k of the fixture's C is this
    // tree's function order[k].
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
    ci_host_ = read_array(case_["arrays"]["ci"].get<std::string>(), {n_det_, nroots_});
    ASSERT_FALSE(HasFailure());

    // The active-space Hamiltonian at C, on the sigma (row-major host arrays
    // need no reorder: h_eff is symmetric and (tu|vw) column-major is (wv|ut)
    // row-major), for the roots' energies.
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

  // The fixture's average as an mcscf::Solution at its orbitals.
  fb::mcscf::Solution solution() {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    const std::size_t block = n_det_ * nroots_;
    fb::mcscf::Solution s{
        .c = fb::SphericalMatrixAO(res, *basis_),
        .roots = {{}, {}, fb::Buffers::device<double>(block, res.device()), {}, 0},
        .energy = 0.0,
        .gradient_norm = 0.0,
        .gamma = {},
        .gamma2 = {},
        .root_gammas = {},
        .iterations = {}};
    upload(res, c_host_, s.c.data());
    upload(res, ci_host_, s.roots.vectors.data());
    fb::Buffers::device<double> hc(block, res.device());
    EXPECT_TRUE(
        sigma_->apply(res, static_cast<std::int32_t>(nroots_), s.roots.vectors.data(), hc.data())
            .has_value());
    const std::vector<double> hc_host = download(res, hc.data(), block);
    for (std::size_t r = 0; r < nroots_; ++r) {
      const auto at = static_cast<std::ptrdiff_t>(r * n_det_);
      const double e = std::inner_product(
          ci_host_.begin() + at, ci_host_.begin() + at + static_cast<std::ptrdiff_t>(n_det_),
          hc_host.begin() + at, 0.0);
      s.roots.energies.push_back(e);
      s.energy += weights_[r] * e;
    }
    auto rdm = fb::ci::Rdm::create(res, *alpha_, *beta_);
    EXPECT_TRUE(rdm) << rdm.error().message;
    fb::Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
    EXPECT_TRUE(
        rdm->rdm12(res, weights_, s.roots.vectors.data(), rdms.data(), rdms.data() + (n * n))
            .has_value());
    s.gamma = download(res, rdms.data(), n * n);
    s.gamma2 = download(res, rdms.data() + (n * n), n * n * n * n);
    for (std::size_t r = 0; r < nroots_; ++r) {
      std::vector<double> one_hot(nroots_, 0.0);
      one_hot[r] = 1.0;
      EXPECT_TRUE(
          rdm->rdm12(res, one_hot, s.roots.vectors.data(), rdms.data(), rdms.data() + (n * n))
              .has_value());
      const std::vector<double> gamma = download(res, rdms.data(), n * n);
      s.root_gammas.insert(s.root_gammas.end(), gamma.begin(), gamma.end());
    }
    return s;
  }

  fb::Result<fb::hessian::SaCasscfTerms>
  run(const fb::mcscf::Solution &s, const std::vector<double> &weights, const std::int32_t root) {
    return fb::hessian::df_sacasscf(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                    *b_, metric_->spherical().data(), s, weights, root, ncore_,
                                    ncas_);
  }

  [[nodiscard]] std::vector<double> root_array(const std::size_t root, const std::string &key,
                                               const std::vector<std::size_t> &shape) const {
    return read_array(case_["roots"][root]["arrays"][key].get<std::string>(), shape);
  }

  // A fixture K (nmo x nmo x count), packed as zvector's z_o.
  std::vector<double> pack(const std::vector<double> &full, const std::size_t count) {
    fb::DeviceResources &res = *res_;
    fb::Buffers::device<double> device(full.size(), res.device());
    upload(res, full, device.data());
    const auto core = static_cast<std::size_t>(ncore_);
    const auto act = static_cast<std::size_t>(ncas_);
    const std::size_t pairs = (core * (nao_ - core)) + (act * (nao_ - core - act));
    fb::Buffers::device<double> packed(pairs * count, res.device());
    for (std::size_t x = 0; x < count; ++x) {
      EXPECT_TRUE(fb::response::pack_orbital_pairs(res, static_cast<std::int32_t>(nao_), ncore_,
                                                   ncas_, device.data() + (nao_ * nao_ * x),
                                                   packed.data() + (pairs * x))
                      .has_value());
    }
    return download(res, packed.data(), pairs * count);
  }

  // Root @p root's Hessian and every term against the fixture's.
  void check_root(const fb::mcscf::Solution &s, const std::size_t root) {
    SCOPED_TRACE(std::format("root {}", root));
    const std::size_t n = n_x_;
    const auto start = std::chrono::steady_clock::now();
    auto got = run(s, weights_, static_cast<std::int32_t>(root));
    ASSERT_TRUE(got) << got.error().message;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::vector<std::size_t> square{n, n};

    std::string report;
    std::vector<std::tuple<std::string, const std::vector<double> *, double>> terms{
        {"exy", &got->exy.exy, kExyTol},
        {"exy_z", &got->z_terms.lxy.exy, kHessianTol},
        {"lxl_lambda", &got->lxlambda_lambda, kHessianTol},
        {"root_hessian_lambda", &got->root_hessian, kHessianTol},
        {"third", &got->third, kHessianTol},
        {"hessian", &got->hessian, kHessianTol},
        {"relaxed_fd", &got->hessian, kRelaxedFdTol}};
    if (equal_) {
      terms.emplace_back("rot", &got->rotation, kHessianTol);
    } else {
      EXPECT_EQ(max_abs(got->rotation), 0.0) << "eq. rot at unequal weights";
    }
    for (const auto &[key, mine, tol] : terms) {
      const std::vector<double> ref = root_array(root, key, square);
      ASSERT_EQ(mine->size(), n * n) << key;
      const double worst = max_diff(*mine, ref);
      EXPECT_LE(worst, tol) << key << ": max |H - ref| " << worst << " of " << max_abs(ref);
      report += std::format("; {} {:.1e} of {:.2e}", key, worst, max_abs(ref));
    }

    // At equal weights, without eq. rot the total is another Hessian.
    if (equal_) {
      std::vector<double> without = got->hessian;
      for (std::size_t i = 0; i < without.size(); ++i) {
        without[i] -= 0.5 * (got->rotation[i] + got->rotation[((i % n) * n) + (i / n)]);
      }
      const double miss = max_diff(without, root_array(root, "relaxed_fd", square));
      EXPECT_GT(miss, kWithoutRotationMiss) << "the total without eq. rot";
      report += std::format("; without rot {:.2e}", miss);
    }

    // z's and lambda^x's orbital parts against the emitter's.
    const std::size_t pairs = got->z.orbital.num_elements();
    const std::vector<double> want_z = pack(root_array(root, "z_orb", {nao_, nao_}), 1);
    const double z_miss = max_diff(download(*res_, got->z.orbital.data(), pairs), want_z);
    EXPECT_LE(z_miss, kSolveTol * std::max(1.0, max_abs(want_z)));
    const std::vector<double> want_lambda =
        pack(read_array(case_["arrays"]["lambda_orb"].get<std::string>(), {nao_, nao_, n}), n);
    const double lambda_miss =
        max_diff(download(*res_, got->cp.orbital.data(), pairs * n), want_lambda);
    EXPECT_LE(lambda_miss, kSolveTol * std::max(1.0, max_abs(want_lambda)));

    ASSERT_EQ(got->residuals.size(), n);
    const double residual = std::ranges::max(got->residuals);
    EXPECT_LE(residual, kResidualTol);
    EXPECT_LE(got->asymmetry, kAsymmetryTol);
    const double drift = sum_rule(got->hessian, n);
    EXPECT_LE(drift, kSumRuleTol);
    std::cout << std::format("[ SA       ] {} root {}: {:.2f} s, CP residual {:.2e} (iterations "
                             "{}), z_orb {:.1e}, lambda_orb {:.1e}, asymmetry {:.1e}, sum rule "
                             "{:.1e}{}\n",
                             case_["name"].get<std::string>(), root, seconds, residual,
                             std::ranges::max(got->cp.iterations), z_miss, lambda_miss,
                             got->asymmetry, drift, report);
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    const fb::mcscf::Solution s = solution();
    ASSERT_FALSE(HasFailure());
    const auto want = case_["energies"].get<std::vector<double>>();
    for (std::size_t r = 0; r < nroots_; ++r) {
      EXPECT_NEAR(s.roots.energies[r], want[r], 1e-9) << "root " << r;
    }
    for (std::size_t root = 0; root < nroots_; ++root) {
      check_root(s, root);
      ASSERT_FALSE(HasFatalFailure());
    }
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
  std::vector<double> weights_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::size_t nroots_ = 0;
  bool equal_ = false;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfSaCasscf, TwoRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa2");
}

TEST_F(HessianDfSaCasscf, TwoRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa2_unequal");
}

TEST_F(HessianDfSaCasscf, ThreeRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa3");
}

TEST_F(HessianDfSaCasscf, ThreeRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa3_unequal");
}

TEST_F(HessianDfSaCasscf, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2_unequal");
  ASSERT_FALSE(HasFatalFailure());
  const fb::mcscf::Solution s = solution();
  ASSERT_FALSE(HasFailure());
  const auto refused = [](const fb::Result<fb::hessian::SaCasscfTerms> &r, const fb::ErrorKind kind,
                          std::string_view what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, kind) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_sacasscf"), std::string::npos) << what;
  };
  const auto config = fb::ErrorKind::InvalidConfig;
  refused(run(s, weights_, 2), config, "root out of range");
  refused(run(s, weights_, -1), config, "negative root");
  refused(run(s, {1.0}, 0), config, "one weight for two roots");
  refused(run(s, {1.0, 0.0}, 0), config, "a zero weight");
  // Weights the solution's RDMs are not the average at.
  refused(run(s, {0.3, 0.7}, 0), config, "the solution's weights swapped");
  // One root: a state-specific solution is df_casscf's.
  fb::mcscf::Solution one = solution();
  one.roots.energies.pop_back();
  refused(run(one, {1.0}, 0), config, "one root");
  // Degenerate roots: the adiabatic energy has no Hessian there.
  fb::mcscf::Solution degenerate = solution();
  degenerate.roots.energies[1] = degenerate.roots.energies[0];
  refused(run(degenerate, weights_, 0), fb::ErrorKind::Numerical, "degenerate roots");
}

} // namespace
