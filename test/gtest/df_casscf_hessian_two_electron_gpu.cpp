// GPU cases for hessian::df_casscf_two_electron, the two-electron explicit
// part of the DF-CASSCF nuclear Hessian (casscf_hessian.tex eq. df2): its
// second-derivative contractions and its J, K and active cross terms against
// data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py, over the same
// cc-pVDZ-JKFIT fit), at the fixture's orbitals and roots, for the stretched
// water CAS(4,4) in 6-31G and in cc-pVDZ (d shells) and one root of a
// two-root state average; their symmetry and the translational sum rule; and
// the refusals.

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
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar against the reference emitter, absolute.
constexpr double kHessianTol = 1e-9;
// Each piece's asymmetry and the total's sum over atoms, absolute, as
// df_rhf_hessian_two_electron_gpu.cpp holds the RHF terms.
constexpr double kSymmetryTol = 1e-12;
constexpr double kSumRuleTol = 1e-9;

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

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
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
                             static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
  }
  return fb::Molecule(res, std::move(atoms));
}

// One case set up as the DF steps would: the bases placed, the fit B and its
// L, the fixture's orbitals made ours (rows over N), and root I's RDMs from
// the fixture's CI vector by ci::Rdm.
class HessianDfCasscfTwoElectron : public ::testing::Test {
protected:
  void load(const std::string &name, const std::size_t root) {
    const auto doc = fb::read_json(kDataDir / "casscf_hessian.json");
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
    const auto nmo = (*c)["nmo"].get<std::size_t>();
    ncore_ = (*c)["ncore"].get<std::int32_t>();
    ncas_ = (*c)["ncas"].get<std::int32_t>();
    const auto nelecas = (*c)["nelecas"].get<std::vector<std::int32_t>>();
    const auto na = (*c)["na"].get<std::size_t>();
    const auto nb = (*c)["nb"].get<std::size_t>();
    const std::size_t nroots = (*c)["roots"].size();
    ASSERT_LT(root, nroots);
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

    // PySCF orders each atom's shells by l (stably), this tree as the BSE
    // file lists them: 6-31G's sp shells interleave s and p here. pyscf[i]
    // is the function here that is PySCF's i-th.
    std::vector<std::size_t> pyscf;
    for (std::uint32_t atom = 0; atom < basis_->natoms(); ++atom) {
      std::vector<std::uint32_t> shells(basis_->atom_shell_count(atom));
      std::iota(shells.begin(), shells.end(), basis_->atom_shell_offset(atom));
      std::ranges::stable_sort(shells, {}, [&](const std::uint32_t s) {
        return basis_->shells()[s].angular_momentum();
      });
      for (const std::uint32_t s : shells) {
        const std::size_t first = basis_->ao_offset_sphe(s);
        for (std::size_t f = 0; f < basis_->shells()[s].function_count_sphe(); ++f) {
          pyscf.push_back(first + f);
        }
      }
    }
    ASSERT_EQ(pyscf.size(), nao_);
    const std::vector<double> n_ao = norms(*basis_);
    const std::vector<double> c_pyscf =
        read_array((*c)["arrays"]["c"].get<std::string>(), {nao_, nmo});
    std::vector<double> c_host(nao_ * nmo);
    for (std::size_t j = 0; j < nmo; ++j) {
      for (std::size_t i = 0; i < nao_; ++i) {
        c_host[pyscf[i] + (nao_ * j)] = c_pyscf[i + (nao_ * j)] / n_ao[pyscf[i]];
      }
    }
    c_.emplace(upload(*res_, c_host));

    // Root I's gamma and Gamma, row-major, from its CI vector (column I of
    // the fixture's n_det x nroots, determinant (ia, ib) at ia nb + ib).
    const std::vector<double> ci =
        read_array((*c)["arrays"]["ci"].get<std::string>(), {na * nb, nroots});
    ASSERT_FALSE(HasFailure());
    const std::vector<double> root_ci(ci.begin() + static_cast<std::ptrdiff_t>(root * na * nb),
                                      ci.begin() +
                                          static_cast<std::ptrdiff_t>((root + 1) * na * nb));
    auto alpha = fb::ci::DeviceStringSet::create(*res_, ncas_, nelecas[0]);
    ASSERT_TRUE(alpha) << alpha.error().message;
    auto beta = fb::ci::DeviceStringSet::create(*res_, ncas_, nelecas[1]);
    ASSERT_TRUE(beta) << beta.error().message;
    auto rdm = fb::ci::Rdm::create(*res_, *alpha, *beta);
    ASSERT_TRUE(rdm) << rdm.error().message;
    ASSERT_EQ(static_cast<std::size_t>(rdm->size()), na * nb);
    const auto n = static_cast<std::size_t>(ncas_);
    const fb::Buffers::device<double> vector = upload(*res_, root_ci);
    fb::Buffers::device<double> gamma(n * n, res_->device());
    fb::Buffers::device<double> gamma2(n * n * n * n, res_->device());
    const std::vector<double> one{1.0};
    ASSERT_TRUE(rdm->rdm12(*res_, one, vector.data(), gamma.data(), gamma2.data()));
    gamma_ = download(*res_, gamma);
    gamma2_ = download(*res_, gamma2);

    const fb::json &arrays = (*c)["roots"][root]["arrays"];
    for (const auto &[key, into] :
         {std::pair{"df2_three_center", &ref_three_center_}, std::pair{"df2_metric", &ref_metric_},
          std::pair{"df2_cross_j", &ref_cross_j_}, std::pair{"df2_cross_k", &ref_cross_k_},
          std::pair{"df2_cross_active", &ref_cross_active_},
          std::pair{"two_electron", &ref_total_}}) {
      *into = read_array(arrays[key].get<std::string>(), {n_x_, n_x_});
    }
  }

  fb::Result<fb::hessian::CasscfTwoElectron>
  run(const fb::BasisSet &aux, const std::int32_t ncore, const std::int32_t ncas,
      const fb::Buffers::device<double> &b, const fb::Buffers::device<double> &c,
      std::span<const double> gamma, std::span<const double> gamma2) {
    return fb::hessian::df_casscf_two_electron(*res_, *basis_, classes_, aux, b,
                                               metric_->spherical().data(), c, ncore, ncas, gamma,
                                               gamma2);
  }

  void check(const std::string &name, const std::size_t root) {
    FBF_REQUIRE_GPU();
    load(name, root);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_FALSE(HasFailure());
    const auto start = std::chrono::steady_clock::now();
    auto got = run(*aux_, ncore_, ncas_, *b_, *c_, gamma_, gamma2_);
    ASSERT_TRUE(got) << got.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // The call's own buffers are freed on return; what stays is the
    // workspace, grown to the largest scratch any of its calls took.
    const std::size_t workspace = res_->device_workspace(0).num_elements();

    std::string report;
    for (const auto &[term, mine, ref] :
         {std::tuple{"three_center", &got->three_center, &ref_three_center_},
          std::tuple{"metric", &got->metric, &ref_metric_},
          std::tuple{"cross_j", &got->cross_coulomb, &ref_cross_j_},
          std::tuple{"cross_k", &got->cross_exchange, &ref_cross_k_},
          std::tuple{"cross_active", &got->cross_active, &ref_cross_active_},
          std::tuple{"total", &got->total, &ref_total_}}) {
      ASSERT_EQ(mine->size(), n_x_ * n_x_) << term;
      const double scale = max_abs(*ref);
      const double worst = max_diff(*mine, *ref);
      const double asym = asymmetry(*mine, n_x_);
      EXPECT_LE(worst, kHessianTol) << term << ": max |H - ref| " << worst << " of " << scale;
      EXPECT_LE(asym, kSymmetryTol) << term;
      report += std::format("; {} max |H| {:.3e}, |H - ref| {:.2e}, asym {:.1e}", term, scale,
                            worst, asym);
    }
    const double drift = sum_rule(got->total, n_x_);
    EXPECT_LE(drift, kSumRuleTol);
    std::cout << std::format("[ hess2e   ] {} root {} ({}): {} coordinates, {:.2f} s, workspace "
                             "{} bytes, total sum rule {:.2e}{}\n",
                             name, root, aux_stamp_, n_x_, elapsed, workspace, drift, report);
  }

  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
  std::vector<double> gamma_;
  std::vector<double> gamma2_;
  std::vector<double> ref_three_center_;
  std::vector<double> ref_metric_;
  std::vector<double> ref_cross_j_;
  std::vector<double> ref_cross_k_;
  std::vector<double> ref_cross_active_;
  std::vector<double> ref_total_;
  std::string aux_stamp_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
};

TEST_F(HessianDfCasscfTwoElectron, Water631gMatchesReference) {
  check("h2o_631g", 0);
}

TEST_F(HessianDfCasscfTwoElectron, WaterCcpvdzMatchesReference) {
  check("h2o_ccpvdz", 0);
}

TEST_F(HessianDfCasscfTwoElectron, StateAverageRootMatchesReference) {
  check("h2o_631g_sa2_unequal", 1);
}

TEST_F(HessianDfCasscfTwoElectron, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g", 0);
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::CasscfTwoElectron> &r,
                                 const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_casscf_two_electron"), std::string::npos) << what;
  };
  const auto nao = static_cast<std::int32_t>(nao_);
  expect_refused(run(*aux_, -1, ncas_, *b_, *c_, gamma_, gamma2_), "ncore -1");
  expect_refused(run(*aux_, ncore_, 0, *b_, *c_, gamma_, gamma2_), "ncas 0");
  expect_refused(run(*aux_, nao - ncas_ + 1, ncas_, *b_, *c_, gamma_, gamma2_),
                 "ncore + ncas past nao");
  const fb::Buffers::device<double> short_c((nao_ * (ncore_ + ncas_)) - 1, res_->device());
  expect_refused(run(*aux_, ncore_, ncas_, *b_, short_c, gamma_, gamma2_),
                 "C shorter than nao nocc");
  const fb::Buffers::device<double> short_b(b_->num_elements() - 1, res_->device());
  expect_refused(run(*aux_, ncore_, ncas_, short_b, *c_, gamma_, gamma2_), "B not naux nao^2");
  expect_refused(
      run(*aux_, ncore_, ncas_, *b_, *c_, std::span{gamma_}.first(gamma_.size() - 1), gamma2_),
      "gamma not ncas^2");
  expect_refused(
      run(*aux_, ncore_, ncas_, *b_, *c_, gamma_, std::span{gamma2_}.first(gamma2_.size() - 1)),
      "Gamma not ncas^4");
  // The fitting set on two of the three atoms.
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  const auto aux_stamp = fb::resolve_aux_basis(aux_stamp_);
  ASSERT_TRUE(aux_stamp);
  const auto aux_two = fb::place(aux_stamp->primitive_normalized(), two);
  ASSERT_TRUE(aux_two);
  expect_refused(run(*aux_two, ncore_, ncas_, *b_, *c_, gamma_, gamma2_), "aux on other atoms");
}

} // namespace
