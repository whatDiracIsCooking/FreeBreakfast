// GPU cases for mcscf::df_derivatives, F^[x] and the CP-CASSCF right-hand
// sides of a DF-CASSCF, against tools/emit/casscf_hessian_reference.py's
// fixtures (test/gtest/data/casscf_hessian/): the orbitals and roots loaded
// from the fixture, never re-converged (PySCF's own solutions of three of the
// cases are saddle points). State-specific in 6-31G and cc-pVDZ (d shells),
// and a state average of two roots at (0.5, 0.5) and (0.7, 0.3) and of three
// at (0.5, 0.3, 0.2): its averaged right-hand sides and each root's own,
// F^[x] at each root's RDMs, and at unequal weights the rotations among the
// roots. Beside them the translational sum rule, the state-specific
// fixed-parameter gradient <c|H~^x|c> + V_nn^x against the fixture's analytic
// gradient, and the refusals.

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
import free_breakfast.ci.strings;
import free_breakfast.mcscf;
import free_breakfast.test.ci_oracles;

namespace {

namespace fb = free_breakfast;

// Against the fixture, absolute: a tenth of the issue's 1e-8. The fixture's
// quantities are analytic over libcint (checked there against finite
// differences), so this is the same formulas over the same fit; measured
// <= 2e-13 but the cc-pVDZ CI part's 2.6e-10 (README.md).
constexpr double kFixtureTol = 1e-9;
// Sum over the atoms of each axis, absolute: translational invariance, which
// none of the terms is built on.
constexpr double kSumRuleTol = 1e-9;
// <c|H~^x|c> + V_nn^x against the fixture's relaxed gradient, which for one
// root is the fixed-parameter one (the parameters are stationary).
constexpr double kGradientTol = 1e-8;

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

std::vector<double> download(fb::DeviceResources &res, const double *device,
                             const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *device) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(device, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
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

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// The largest |Sum_A X[..., 3 A + k]| over k and every element of a stack of
// n_atom * 3 slabs of @p len doubles.
double sum_rule(const std::vector<double> &stack, const std::size_t len, const std::size_t n_atom) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t i = 0; i < len; ++i) {
      double sum = 0.0;
      for (std::size_t a = 0; a < n_atom; ++a) {
        sum += stack[(((3 * a) + k) * len) + i];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

// The singlet part of every length-@p dim column of @p v, in place:
// prod_{S = 1..S_max} (S^2 - S(S + 1)) / (0 - S(S + 1)), over the host S^2.
void project_singlet(const fb::ci::StringSet &alpha, const fb::ci::StringSet &beta,
                     std::vector<double> &v, const std::size_t dim) {
  const std::int32_t s_max = (alpha.n_electrons() + beta.n_electrons()) / 2;
  std::vector<double> s2c(dim);
  for (std::size_t at = 0; at < v.size(); at += dim) {
    const std::span<double> col(v.data() + at, dim);
    for (std::int32_t s = 1; s <= s_max; ++s) {
      const double eig = s * (s + 1.0);
      ASSERT_TRUE(fb::test::host_spin_square(alpha, beta, col, s2c));
      for (std::size_t i = 0; i < dim; ++i) {
        col[i] = (s2c[i] - (eig * col[i])) / -eig;
      }
    }
  }
}

class McscfDerivative : public ::testing::Test {
protected:
  // One case set up as the DF steps would: the bases placed, the core
  // Hamiltonian, the fit B and its L, the fixture's orbitals made ours (rows
  // over N) and its roots.
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

    std::vector<double> norms;
    for (const fb::Shell &shell : basis_->shells()) {
      norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
    }
    // PySCF orders each atom's shells by l, keeping the file's order within
    // one l; this tree keeps the file's order, so a split SP shell (6-31G's)
    // puts 2p before 3s here and after it there. Row k of the fixture's C is
    // this tree's function order[k].
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
    const std::vector<double> c_pyscf = read_array(array("c"), {nao_, nao_});
    std::vector<double> c(nao_ * nao_);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t k = 0; k < nao_; ++k) {
        c[order[k] + (nao_ * j)] = c_pyscf[k + (nao_ * j)] / norms[order[k]];
      }
    }
    c_.emplace(*res_, *basis_);
    upload(*res_, c, c_->data());
    const std::vector<double> ci = read_array(array("ci"), {n_det_, weights_.size()});
    roots_.emplace(ci.size(), res_->device());
    upload(*res_, ci, roots_->data());
  }

  [[nodiscard]] std::string array(const std::string &key) const {
    return case_["arrays"][key].get<std::string>();
  }
  [[nodiscard]] std::string root_array(const std::size_t root, const std::string &key) const {
    return case_["roots"][root]["arrays"][key].get<std::string>();
  }

  fb::Result<fb::mcscf::Derivatives> run(std::span<const double> weights) {
    return fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                     *b_, metric_->spherical().data(), *c_, ncore_, ncas_, nalpha_,
                                     nbeta_, roots_->data(), weights);
  }

  // The three device stacks on the host, their sum rules checked.
  struct Host {
    std::vector<double> fock;
    std::vector<double> orbital;
    std::vector<double> ci;
  };
  Host to_host(const fb::mcscf::Derivatives &d, const std::string &what) {
    const std::size_t nn = nao_ * nao_;
    const std::size_t nr = weights_.size();
    Host h{download(*res_, d.fock.data(), nn * n_x_),
           download(*res_, d.rhs_orbital.data(), nn * n_x_),
           download(*res_, d.rhs_ci.data(), n_det_ * nr * n_x_)};
    const std::size_t n_atom = n_x_ / 3;
    EXPECT_LT(sum_rule(h.fock, nn, n_atom), kSumRuleTol) << what << " F^[x]";
    EXPECT_LT(sum_rule(h.orbital, nn, n_atom), kSumRuleTol) << what << " rhs_orb";
    EXPECT_LT(sum_rule(h.ci, n_det_ * nr, n_atom), kSumRuleTol) << what << " rhs_ci";
    EXPECT_LT(sum_rule(d.rhs_theta, nr * nr, n_atom), kSumRuleTol) << what << " rhs_theta";
    return h;
  }

  // The averaged right-hand sides at the case's weights, and per root its own
  // (one-hot weights over the same roots) and F^[x] at its RDMs.
  void check() {
    const std::size_t nr = weights_.size();
    // The fixture's roots carry a quintet part (6e-7 in h2o_631g's) and its
    // CI covectors are over the singlet space (data/README.md), so the CI
    // parts are compared as their singlet parts.
    const auto alpha = fb::ci::StringSet::create(ncas_, nalpha_);
    const auto beta = fb::ci::StringSet::create(ncas_, nbeta_);
    ASSERT_TRUE(alpha && beta);
    const bool equal = case_["equal_weights"].get<bool>();
    const std::vector<std::size_t> square{nao_, nao_, n_x_};
    const std::vector<std::size_t> ci_shape{n_det_, nr, n_x_};
    const std::vector<std::size_t> theta_shape{nr, nr, n_x_};
    {
      auto d = run(weights_);
      ASSERT_TRUE(d) << d.error().message;
      Host h = to_host(*d, "average");
      project_singlet(*alpha, *beta, h.ci, n_det_);
      const std::vector<double> orb = read_array(array("rhs_orb"), square);
      const std::vector<double> ci = read_array(array("rhs_ci"), ci_shape);
      EXPECT_LT(max_diff(h.orbital, orb), kFixtureTol) << "max |rhs_orb| " << max_abs(orb);
      EXPECT_LT(max_diff(h.ci, ci), kFixtureTol) << "max |rhs_ci| " << max_abs(ci);
      if (!equal) {
        const std::vector<double> theta = read_array(array("rhs_theta"), theta_shape);
        EXPECT_LT(max_diff(d->rhs_theta, theta), kFixtureTol);
      }
      if (nr == 1) {
        // <c|H~^x|c> + V_nn^x is the state-specific gradient.
        const std::vector<double> vnn = fb::nuclear_repulsion_gradient(*molecule_);
        const auto want = case_["roots"][0]["gradient"].get<std::vector<double>>();
        ASSERT_EQ(want.size(), n_x_);
        double worst = 0.0;
        for (std::size_t x = 0; x < n_x_; ++x) {
          worst = std::max(worst, std::abs(d->hamiltonian[x] + vnn[x] - want[x]));
        }
        EXPECT_LT(worst, kGradientTol);
      }
    }
    for (std::size_t root = 0; root < nr; ++root) {
      std::vector<double> one_hot(nr, 0.0);
      one_hot[root] = 1.0;
      auto d = run(one_hot);
      ASSERT_TRUE(d) << d.error().message;
      Host h = to_host(*d, std::format("root {}", root));
      project_singlet(*alpha, *beta, h.ci, n_det_);
      // fock_x is written x first, (n, nmo, nmo): element (x, p, q) at
      // x + n (p + nmo q).
      const std::vector<double> fock_xpq =
          read_array(root_array(root, "fock_x"), {n_x_, nao_, nao_});
      std::vector<double> fock_x(fock_xpq.size());
      for (std::size_t x = 0; x < n_x_; ++x) {
        for (std::size_t pq = 0; pq < nao_ * nao_; ++pq) {
          fock_x[pq + (nao_ * nao_ * x)] = fock_xpq[x + (n_x_ * pq)];
        }
      }
      EXPECT_LT(max_diff(h.fock, fock_x), kFixtureTol) << "root " << root;
      if (nr == 1) {
        continue;
      }
      const std::vector<double> orb = read_array(root_array(root, "root_rhs_orb"), square);
      const std::vector<double> ci = read_array(root_array(root, "root_rhs_ci"), ci_shape);
      EXPECT_LT(max_diff(h.orbital, orb), kFixtureTol) << "root " << root;
      EXPECT_LT(max_diff(h.ci, ci), kFixtureTol) << "root " << root;
      if (!equal) {
        const std::vector<double> theta =
            read_array(root_array(root, "root_rhs_theta"), theta_shape);
        EXPECT_LT(max_diff(d->rhs_theta, theta), kFixtureTol) << "root " << root;
      }
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
  std::optional<fb::SphericalMatrixAO> c_;
  std::optional<fb::Buffers::device<double>> roots_;
  std::vector<double> weights_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(McscfDerivative, StateSpecificMatchesReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g");
  check();
}

// d shells, one of them active.
TEST_F(McscfDerivative, StateSpecificDShellsMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_ccpvdz");
  check();
}

TEST_F(McscfDerivative, TwoRootsEqualMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2");
  check();
}

TEST_F(McscfDerivative, TwoRootsUnequalMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2_unequal");
  check();
}

TEST_F(McscfDerivative, ThreeRootsUnequalMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3_unequal");
  check();
}

TEST_F(McscfDerivative, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g");
  const std::vector<double> one{1.0};
  const auto expect_config = [](const auto &r, std::string_view what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("mcscf::df_derivatives"), std::string::npos) << what;
  };
  expect_config(fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_,
                                          h_->spherical(), *b_, metric_->spherical().data(), *c_,
                                          ncore_, 0, nalpha_, nbeta_, roots_->data(), one),
                "ncas 0");
  expect_config(fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_,
                                          h_->spherical(), *b_, metric_->spherical().data(), *c_,
                                          -1, ncas_, nalpha_, nbeta_, roots_->data(), one),
                "negative ncore");
  expect_config(fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_,
                                          h_->spherical(), *b_, metric_->spherical().data(), *c_,
                                          ncore_, static_cast<std::int32_t>(nao_), nalpha_, nbeta_,
                                          roots_->data(), one),
                "ncore + ncas past nao");
  expect_config(fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_,
                                          h_->spherical(), *b_, metric_->spherical().data(), *c_,
                                          ncore_, ncas_, nalpha_, nbeta_, roots_->data(), {}),
                "no roots");
  const fb::Buffers::device<double> short_b(naux_, res_->device());
  expect_config(fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_,
                                          h_->spherical(), short_b, metric_->spherical().data(),
                                          *c_, ncore_, ncas_, nalpha_, nbeta_, roots_->data(), one),
                "B's length");
}

} // namespace
