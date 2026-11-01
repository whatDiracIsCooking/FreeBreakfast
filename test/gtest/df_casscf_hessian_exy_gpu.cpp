// GPU cases for hessian::df_casscf_exy, the fixed-parameter second derivative
// E_xy of a DF-CASSCF energy along the Lowdin connection (casscf_hessian.tex
// eq. exy): each of its pieces (the explicit one-electron, V_nn and
// two-electron parts and the connection's three terms) and their sum against
// data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py, over the same
// cc-pVDZ-JKFIT fit), at the fixture's orbitals and roots, with F^[x] from
// mcscf::df_derivatives at the same root: the stretched water CAS(4,4) in
// 6-31G and in cc-pVDZ (d shells), and one root of a two-root and of a
// three-root state average; that dropping tau^xy's 3/8 (S^x S^y + S^y S^x)
// misses the reference; symmetry and the translational sum rule; and the
// refusals.

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
import free_breakfast.mcscf;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar against the reference emitter, absolute, per piece and for
// the total.
constexpr double kHessianTol = 1e-8;
// Each piece's asymmetry and E_xy's sum over atoms, absolute.
constexpr double kSymmetryTol = 1e-11;
constexpr double kSumRuleTol = 1e-8;

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

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
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

// One case set up as the DF steps would: the bases placed, h, the fit B and
// its L, the fixture's orbitals made ours (rows over N, in this tree's AO
// order), its roots on the device, and root I's RDMs by ci::Rdm.
class HessianDfCasscfExy : public ::testing::Test {
protected:
  void load(const std::string &name, const std::size_t root) {
    const auto doc = fb::read_json(kDataDir / "casscf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        case_ = entry;
      }
    }
    ASSERT_FALSE(case_.is_null()) << "no case " << name;
    res_.emplace(0);
    std::vector<fb::Atom> atoms;
    for (const auto &atom : case_["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                               static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
    }
    molecule_.emplace(*res_, std::move(atoms));
    const auto stamp = fb::resolve_basis(case_["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    aux_stamp_ = case_["aux_basis"].get<std::string>();
    const auto aux_stamp = fb::resolve_aux_basis(aux_stamp_);
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
    nroots_ = case_["weights"].size();
    ASSERT_LT(root, nroots_);
    root_ = root;
    n_det_ = case_["na"].get<std::size_t>() * case_["nb"].get<std::size_t>();

    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto su = fb::upload_spherical_transform_matrix(*res_, *basis_);
    ASSERT_TRUE(su) << su.error().message;
    h_.emplace(*res_, *basis_, std::move(*su));
    ASSERT_TRUE(fb::core_hamiltonian(*res_, classes_, *molecule_, h_->cartesian()));
    ASSERT_TRUE(h_->cartesian_to_spherical(*res_));
    auto u = fb::upload_spherical_transform_matrix(*res_, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(*res_, *aux_, std::move(*u));
    b_.emplace(naux_ * nao_ * nao_, res_->device());
    ASSERT_TRUE(fb::metric(*res_, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(*res_, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(*res_, static_cast<std::int32_t>(nao_),
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
    c_.emplace(upload(*res_, c_host_));
    c_matrix_.emplace(*res_, *basis_);
    ASSERT_EQ(wwr::wwrMemcpyAsync(c_matrix_->data(), c_host_.data(),
                                  sizeof(double) * c_host_.size(), wwr::wwrMemcpyHostToDevice,
                                  res_->stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(res_->sync());

    // Every root on the device (determinant (ia, ib) at ia nb + ib, root r
    // at r n_det), and root I's gamma and Gamma, row-major.
    const std::vector<double> ci =
        read_array(case_["arrays"]["ci"].get<std::string>(), {n_det_, nroots_});
    ASSERT_FALSE(HasFailure());
    roots_.emplace(upload(*res_, ci));
    auto alpha = fb::ci::DeviceStringSet::create(*res_, ncas_, nalpha_);
    ASSERT_TRUE(alpha) << alpha.error().message;
    auto beta = fb::ci::DeviceStringSet::create(*res_, ncas_, nbeta_);
    ASSERT_TRUE(beta) << beta.error().message;
    auto rdm = fb::ci::Rdm::create(*res_, *alpha, *beta);
    ASSERT_TRUE(rdm) << rdm.error().message;
    ASSERT_EQ(static_cast<std::size_t>(rdm->size()), n_det_);
    const auto n = static_cast<std::size_t>(ncas_);
    fb::Buffers::device<double> gamma(n * n, res_->device());
    fb::Buffers::device<double> gamma2(n * n * n * n, res_->device());
    const std::vector<double> one{1.0};
    ASSERT_TRUE(
        rdm->rdm12(*res_, one, roots_->data() + (root * n_det_), gamma.data(), gamma2.data()));
    gamma_ = download(*res_, gamma.data(), gamma.num_elements());
    gamma2_ = download(*res_, gamma2.data(), gamma2.num_elements());

    // F^[x] at root I's RDMs: the derivatives at weights one-hot on I.
    std::vector<double> one_hot(nroots_, 0.0);
    one_hot[root] = 1.0;
    auto derivatives =
        fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(), *b_,
                                  metric_->spherical().data(), *c_matrix_, ncore_, ncas_, nalpha_,
                                  nbeta_, roots_->data(), one_hot);
    ASSERT_TRUE(derivatives) << derivatives.error().message;
    fock_x_.emplace(std::move(derivatives->fock));
  }

  [[nodiscard]] std::vector<double> root_array(const std::string &key,
                                               const std::vector<std::size_t> &shape) const {
    return read_array(case_["roots"][root_]["arrays"][key].get<std::string>(), shape);
  }

  fb::Result<fb::hessian::CasscfExy>
  run(const fb::BasisSet &aux, const std::int32_t ncore, const std::int32_t ncas,
      const fb::Buffers::device<double> &c, std::span<const double> gamma,
      std::span<const double> gamma2, const fb::Buffers::device<double> &fock_x) {
    return fb::hessian::df_casscf_exy(*res_, *molecule_, *basis_, classes_, aux, h_->spherical(),
                                      *b_, metric_->spherical().data(), c, ncore, ncas, gamma,
                                      gamma2, fock_x);
  }

  // 2 Sum F (3/8) (S^x S^y + S^y S^x) on the host, from the fixture's F and
  // S^x = C^T dS/dx C over this tree's AO S^x: the part of tau^xy the test
  // that drops it takes out.
  std::vector<double> three_eighths_term() {
    const auto s_ao = fb::overlap_derivative(*res_, *basis_, classes_);
    EXPECT_TRUE(s_ao) << s_ao.error().message;
    if (!s_ao) {
      return {};
    }
    const std::size_t nn = nao_ * nao_;
    const std::vector<double> s = download(*res_, s_ao->data(), nn * n_x_);
    const std::vector<double> f = root_array("fock", {nao_, nao_});
    std::vector<std::vector<double>> s_mo(n_x_, std::vector<double>(nn, 0.0));
    std::vector<double> half(nn);
    for (std::size_t x = 0; x < n_x_; ++x) {
      const double *const sx = s.data() + (x * nn);
      // half = S^x C, then C^T half.
      for (std::size_t q = 0; q < nao_; ++q) {
        for (std::size_t m = 0; m < nao_; ++m) {
          double sum = 0.0;
          for (std::size_t k = 0; k < nao_; ++k) {
            sum += sx[m + (nao_ * k)] * c_host_[k + (nao_ * q)];
          }
          half[m + (nao_ * q)] = sum;
        }
      }
      for (std::size_t q = 0; q < nao_; ++q) {
        for (std::size_t p = 0; p < nao_; ++p) {
          double sum = 0.0;
          for (std::size_t m = 0; m < nao_; ++m) {
            sum += c_host_[m + (nao_ * p)] * half[m + (nao_ * q)];
          }
          s_mo[x][p + (nao_ * q)] = sum;
        }
      }
    }
    std::vector<double> term(n_x_ * n_x_);
    for (std::size_t y = 0; y < n_x_; ++y) {
      for (std::size_t x = 0; x < n_x_; ++x) {
        double sum = 0.0;
        for (std::size_t q = 0; q < nao_; ++q) {
          for (std::size_t p = 0; p < nao_; ++p) {
            double product = 0.0;
            for (std::size_t r = 0; r < nao_; ++r) {
              product += (s_mo[x][p + (nao_ * r)] * s_mo[y][r + (nao_ * q)]) +
                         (s_mo[y][p + (nao_ * r)] * s_mo[x][r + (nao_ * q)]);
            }
            sum += f[p + (nao_ * q)] * product;
          }
        }
        term[x + (n_x_ * y)] = 2.0 * 0.375 * sum;
      }
    }
    return term;
  }

  void check(const std::string &name, const std::size_t root) {
    FBF_REQUIRE_GPU();
    load(name, root);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_FALSE(HasFailure());
    const auto start = std::chrono::steady_clock::now();
    auto got = run(*aux_, ncore_, ncas_, *c_, gamma_, gamma2_, *fock_x_);
    ASSERT_TRUE(got) << got.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::vector<std::size_t> square{n_x_, n_x_};

    std::string report;
    for (const auto &[term, key, mine] :
         {std::tuple{"one_electron", "one_electron", &got->one_electron},
          std::tuple{"nuclear_repulsion", "nuclear_repulsion", &got->nuclear_repulsion},
          std::tuple{"two_electron", "two_electron", &got->two_electron.total},
          std::tuple{"connection_second", "connection_second", &got->connection_second},
          std::tuple{"connection_fock", "connection_fock", &got->connection_fock},
          std::tuple{"connection_twice", "connection_twice", &got->connection_twice},
          std::tuple{"exy", "exy", &got->exy}}) {
      const std::vector<double> ref = root_array(key, square);
      ASSERT_EQ(mine->size(), n_x_ * n_x_) << term;
      const double scale = max_abs(ref);
      const double worst = max_diff(*mine, ref);
      const double asym = asymmetry(*mine, n_x_);
      EXPECT_LE(worst, kHessianTol) << term << ": max |H - ref| " << worst << " of " << scale;
      EXPECT_LE(asym, kSymmetryTol) << term;
      report += std::format("; {} max |H| {:.3e}, |H - ref| {:.2e}, asym {:.1e}", term, scale,
                            worst, asym);
    }
    const double drift = sum_rule(got->exy, n_x_);
    EXPECT_LE(drift, kSumRuleTol);
    std::cout << std::format("[ exy      ] {} root {} ({}): {} coordinates, {:.2f} s, E_xy sum "
                             "rule {:.2e}{}\n",
                             name, root, aux_stamp_, n_x_, elapsed, drift, report);
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
  std::optional<fb::Buffers::device<double>> c_;
  std::optional<fb::SphericalMatrixAO> c_matrix_;
  std::optional<fb::Buffers::device<double>> roots_;
  std::optional<fb::Buffers::device<double>> fock_x_;
  std::vector<double> c_host_;
  std::vector<double> gamma_;
  std::vector<double> gamma2_;
  std::string aux_stamp_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::size_t nroots_ = 0;
  std::size_t root_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfCasscfExy, Water631gMatchesReference) {
  check("h2o_631g", 0);
}

TEST_F(HessianDfCasscfExy, WaterCcpvdzMatchesReference) {
  check("h2o_ccpvdz", 0);
}

TEST_F(HessianDfCasscfExy, TwoRootAverageRootMatchesReference) {
  check("h2o_631g_sa2_unequal", 1);
}

TEST_F(HessianDfCasscfExy, ThreeRootAverageRootMatchesReference) {
  check("h2o_631g_sa3_unequal", 2);
}

// tau^xy's 3/8 (S^x S^y + S^y S^x) is what keeps the connection orthonormal
// at second order. Taken out of E_xy, the rest misses the reference by about
// E_xy's own size (the design's table: 0.93 against finite differences),
// while the term itself is the size of that miss.
TEST_F(HessianDfCasscfExy, DroppingTheThreeEighthsTermsMisses) {
  FBF_REQUIRE_GPU();
  load("h2o_631g", 0);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_FALSE(HasFailure());
  auto got = run(*aux_, ncore_, ncas_, *c_, gamma_, gamma2_, *fock_x_);
  ASSERT_TRUE(got) << got.error().message;
  const std::vector<double> term = three_eighths_term();
  ASSERT_EQ(term.size(), n_x_ * n_x_);
  const std::vector<double> ref = root_array("exy", {n_x_, n_x_});
  std::vector<double> without(got->exy);
  for (std::size_t i = 0; i < without.size(); ++i) {
    without[i] -= term[i];
  }
  const double scale = max_abs(ref);
  const double with_miss = max_diff(got->exy, ref);
  const double without_miss = max_diff(without, ref);
  EXPECT_LE(with_miss, kHessianTol);
  EXPECT_GT(without_miss, 0.1 * scale);
  EXPECT_NEAR(without_miss, max_abs(term), kHessianTol);
  std::cout << std::format("[ exy      ] h2o_631g without the 3/8 terms: max |E_xy - ref| {:.3e} "
                           "of max |E_xy| {:.3e} (with them {:.2e})\n",
                           without_miss, scale, with_miss);
}

TEST_F(HessianDfCasscfExy, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g", 0);
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::CasscfExy> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_casscf_exy"), std::string::npos) << what;
  };
  const auto nao = static_cast<std::int32_t>(nao_);
  expect_refused(run(*aux_, -1, ncas_, *c_, gamma_, gamma2_, *fock_x_), "ncore -1");
  expect_refused(run(*aux_, ncore_, 0, *c_, gamma_, gamma2_, *fock_x_), "ncas 0");
  expect_refused(run(*aux_, nao - ncas_ + 1, ncas_, *c_, gamma_, gamma2_, *fock_x_),
                 "ncore + ncas past nmo");
  const fb::Buffers::device<double> ragged_c((nao_ * nao_) - 1, res_->device());
  expect_refused(run(*aux_, ncore_, ncas_, ragged_c, gamma_, gamma2_, *fock_x_),
                 "C not a whole number of columns");
  expect_refused(
      run(*aux_, ncore_, ncas_, *c_, std::span{gamma_}.first(gamma_.size() - 1), gamma2_, *fock_x_),
      "gamma not ncas^2");
  expect_refused(run(*aux_, ncore_, ncas_, *c_, gamma_,
                     std::span{gamma2_}.first(gamma2_.size() - 1), *fock_x_),
                 "Gamma not ncas^4");
  const fb::Buffers::device<double> short_fock(fock_x_->num_elements() - 1, res_->device());
  expect_refused(run(*aux_, ncore_, ncas_, *c_, gamma_, gamma2_, short_fock),
                 "F^[x] not 3 n_atom nmo^2");
  // The fitting set on two of the three atoms.
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  const auto aux_stamp = fb::resolve_aux_basis(aux_stamp_);
  ASSERT_TRUE(aux_stamp);
  const auto aux_two = fb::place(aux_stamp->primitive_normalized(), two);
  ASSERT_TRUE(aux_two);
  expect_refused(run(*aux_two, ncore_, ncas_, *c_, gamma_, gamma2_, *fock_x_),
                 "aux on other atoms");
}

} // namespace
