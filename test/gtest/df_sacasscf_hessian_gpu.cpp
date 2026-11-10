// GPU cases for hessian::df_sacasscf, a state-averaged DF-CASSCF root's
// analytic nuclear Hessian (docs/hessian/hessian.tex eq. sa), against
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
// total without eq. rot misses relaxed_fd; and the refusals, among them a
// state average on a continuous family of solutions, HF CAS(10,7) at two
// roots, this tree's own DF-CASSCF through the registry (#997). And
// df_sacasscf_footprint (#1176), at equal and unequal weights.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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
import free_breakfast.recipes.diagonalize;
import free_breakfast.hessian;
import free_breakfast.io.base;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "pool_footprint.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

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
  FBF_SKIP_IF_FAST_TESTS();
  check("h2o_631g_sa2");
}

TEST_F(HessianDfSaCasscf, TwoRootsUnequalWeightsMatchReference) {
  FBF_FAMILY_REPRESENTATIVE();
  check("h2o_631g_sa2_unequal");
}

TEST_F(HessianDfSaCasscf, ThreeRootsEqualWeightsMatchReference) {
  FBF_SKIP_IF_FAST_TESTS();
  check("h2o_631g_sa3");
}

TEST_F(HessianDfSaCasscf, ThreeRootsUnequalWeightsMatchReference) {
  FBF_SKIP_IF_FAST_TESTS();
  check("h2o_631g_sa3_unequal");
}

// df_sacasscf's own allocations held to df_sacasscf_footprint at its least
// (H~'s block products at one column's J and K, the derivative integrals at
// one coordinate a chunk): on a fresh DeviceResources whose pool is capped at
// what a probe run leaves held (the workspace), the fragmentation headroom an
// adaptive part leaves of the budget, and the footprint's peak; run once to
// grow the workspace, then measured, the pool never past the cap. Root 1 of
// the two-root average at equal weights (eq. rot's root_rotation runs) and at
// unequal ones (the rotations are parameters); four runs of 2.5 s each, so
// slow-gated: the analytic CASSCF [hessian]'s footprint
// (pipeline_step_footprint_gpu.cpp) stands for this in the fast tier.
TEST_F(HessianDfSaCasscf, FootprintIsSoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  for (const std::string name : {"h2o_631g_sa2", "h2o_631g_sa2_unequal"}) {
    SCOPED_TRACE(name);
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    const fb::mcscf::Solution s = solution();
    ASSERT_FALSE(HasFailure());
    const bool rotating = std::ranges::contains(fb::response::rotating_pairs(weights_), true);
    ASSERT_EQ(rotating, name.ends_with("unequal"));
    const std::size_t lwork = fb::syevd_lwork<double>(*res_, ncas_).value();
    const fb::MemoryFootprint fp =
        fb::hessian::df_sacasscf_footprint(*basis_, *aux_, ncore_, ncas_, nalpha_, nbeta_,
                                           static_cast<std::int32_t>(nroots_), rotating, lwork);
    const auto call = [&](fb::DeviceResources &res) {
      auto got =
          fb::hessian::df_sacasscf(res, *molecule_, *basis_, classes_, *aux_, h_->spherical(), *b_,
                                   metric_->spherical().data(), s, weights_, 1, ncore_, ncas_);
      ASSERT_TRUE(got) << got.error().message;
    };
    std::size_t held = 0;
    {
      fb::DeviceResources probe{0};
      call(probe);
      ASSERT_TRUE(probe.sync());
      EXPECT_EQ(probe.device_workspace(0).num_elements(), fp.device.workspace);
      held = probe.device()->pool_usage().used_current;
    }
    const std::uint64_t cap =
        held + fb::kPoolHeadroomBytes + fp.device.persistent + fp.device.scratch;
    fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
    call(res);
    const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] { call(res); });
    std::cout << std::format("[ footprint ] {} root 1 at the least, capped at {}: measured {}, "
                             "footprint {} + workspace {}\n",
                             name, cap, peak.used, fp.device.persistent + fp.device.scratch,
                             fp.device.workspace);
    // At the peak: the strings' four, the integrals', the Fock's, the
    // sigma's, H's and H~'s members, and a phase's own.
    fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                          64);
    EXPECT_LE(res.device()->pool_usage().used_high, cap);
  }
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

// A state average on a continuous family of solutions (#997): HF in cc-pVDZ
// with CAS(4,3), 3sigma, one orbital of the 1pi pair and sigma* active and
// the other 1pi orbital core, averaged over two roots at equal weights.
// Rotating every orbital about the bond, each root following, leaves every
// root's energy and so E_SA unchanged, and with one 1pi orbital active it is
// no rotation inside the active space: A has a null direction, which the
// average does not lift. df_sacasscf's probe column refuses it.
//
// Not CAS(10,7), one orbital of the 2pi pair active (#1188): its second root
// is one component of the 1Pi pair, the average splits the pair, and its
// CASSCF converged erratically, in 12 to 50 macro iterations and in none in
// 2 of 40 runs (src/mcscf/solve/README.md, "A state average that splits a
// degenerate pair"). Over three roots that pair is degenerate, which
// df_sacasscf refuses before its probe. In CAS(4,3) the other component's
// 1pi is core, so no pair is split, and the CASSCF takes 6.
TEST_F(HessianDfSaCasscf, ContinuousFamilyIsRefused) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("df_sacasscf_hessian_family");
  ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", "2\nhf\nH 0 0 0\nF 0 0 0.917\n").has_value());
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = "
                             "\"cc-pvdz\"\n[eri]\nmode = \"density_fitting\"\naux_basis = "
                             "\"cc-pVDZ-JKFIT\"\n[casscf]\nncas = 3\nnelecas = 4\nactive = "
                             "[2, 3, 5]\nnroots = 2\nconv_tol = 1e-12\n"
                             "conv_tol_grad = 1e-10\n")
                  .has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  fb::DeviceResources &res = node.device(0);
  const auto &cas = calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  ASSERT_EQ(cas->weights.size(), 2U);
  EXPECT_LE(cas->iterations.size(), 20U);
  const fb::BasisSet &basis = calc->get<fb::BasisSetOutput>()->basis;
  const auto &fit = std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  // The Solution as the [hessian] step builds it: gamma the weighted average
  // of the roots' 1-RDMs, rdm2 already the average.
  const auto n2 = static_cast<std::size_t>(cas->ncas) * static_cast<std::size_t>(cas->ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas->weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas->weights[r] * cas->rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution s{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = {},
      .weights = cas->weights};
  upload(res, cas->vectors, s.roots.vectors.data());
  ASSERT_EQ(wwr::wwrMemcpyAsync(s.c.data(), cas->c.data(), cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  std::cout << std::format(
      "[ casscf   ] hf CAS(4,3) SA-2: E {:.10f} and {:.10f}, ||g|| {:.1e} in {} macro "
      "iterations\n",
      cas->energies[0], cas->energies[1], cas->gradient_norm, cas->iterations.size());

  const auto h = fb::hessian::df_sacasscf(res, calc->get<fb::MoleculeOutput>()->molecule, basis,
                                          calc->get<fb::PrimitivePairClassesOutput>()->classes,
                                          fit.aux, calc->get<fb::CoreHamiltonianOutput>()->h, fit.b,
                                          fit.l.data(), s, cas->weights, 0, cas->ncore, cas->ncas);
  ASSERT_FALSE(h.has_value()) << "a state average with a null direction was differentiated";
  const std::string &m = h.error().message;
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Divergence) << m;
  EXPECT_NE(m.find("hessian::df_sacasscf: the CP-CASSCF's H~ is singular or not positive"),
            std::string::npos)
      << m;
  // When the probe's column finished, its rho, under the bar it names (it can
  // instead meet a direction H~ is not positive along, or run out of
  // products: the same refusal).
  const std::string_view rho_at = "Rayleigh quotient ";
  const std::string_view bar_at = "not above the ";
  if (const std::size_t at = m.find(rho_at); at != std::string::npos) {
    const std::size_t bar = m.find(bar_at, at);
    ASSERT_NE(bar, std::string::npos) << m;
    EXPECT_LT(std::stod(m.substr(at + rho_at.size())), std::stod(m.substr(bar + bar_at.size())))
        << m;
  }
  std::cout << std::format("[ family   ] {}\n", m.substr(0, 400));
}

} // namespace
