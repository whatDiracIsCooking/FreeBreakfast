// mcscf::df_gradient's state-average weights (#608), over
// tools/emit/casscf_hessian_reference.py's stretched water, CAS(4,4) /
// 6-31G / cc-pVDZ-JKFIT (test/gtest/data/casscf_hessian/): the orbitals and
// roots loaded from the fixture, never re-converged (PySCF's own two-root
// solution is a saddle point), into a Solution with the RDMs at the case's
// weights. Two roots at (0.7, 0.3), three at equal weights and three at
// (0.5, 0.3, 0.2): every root's gradient against the fixture's (the exact
// parameter gradient, checked there against finite differences). At three
// unequal weights the Z-vector carries the rotations among the roots (#613);
// projected out, as before it, the gradient is off by up to 3.3e-3.

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

namespace {

namespace fb = free_breakfast;

// Absolute, against the fixture's exact parameter gradient over the same fit
// (itself good to 6e-9 against finite differences, its orbitals converged to
// 2e-14). Measured on the gfx1200: at most 7e-12 over every root of the
// three cases.
constexpr double kGradientTol = 1e-9;

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

class McscfGradientWeights : public ::testing::Test {
protected:
  // One case set up as the DF steps would: the bases placed, the core
  // Hamiltonian, the fit B and its L, the fixture's orbitals made ours (rows
  // over N, in this tree's AO order) and its roots (mcscf_root_rotation_gpu.cpp's).
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
    ncore_ = case_["ncore"].get<std::int32_t>();
    ncas_ = case_["ncas"].get<std::int32_t>();
    const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
    nalpha_ = nelecas[0];
    nbeta_ = nelecas[1];
    nroots_ = case_["weights"].size();
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
    // one l; this tree keeps the file's order, so 6-31G's split SP shell puts
    // 2p before 3s here and after it there. Row k of the fixture's C is this
    // tree's function order[k].
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
    c_.assign(nao_ * nao_, 0.0);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t k = 0; k < nao_; ++k) {
        c_[order[k] + (nao_ * j)] = c_pyscf[k + (nao_ * j)] / norms[order[k]];
      }
    }
    ci_ = read_array(case_["arrays"]["ci"].get<std::string>(), {n_det_, nroots_});
  }

  // The fixture's orbitals and roots as an mcscf::Solution whose RDMs are at
  // @p weights: each root's gamma, and the weighted gamma and Gamma, by
  // ci::Rdm over the roots. Converged only at the case's own weights.
  fb::mcscf::Solution solution(const std::vector<double> &weights) {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    fb::SphericalMatrixAO c(res, *basis_);
    upload(res, c_, c.data());
    fb::Buffers::device<double> vectors(ci_.size(), res.device());
    upload(res, ci_, vectors.data());
    auto alpha = fb::ci::DeviceStringSet::create(res, ncas_, nalpha_);
    auto beta = fb::ci::DeviceStringSet::create(res, ncas_, nbeta_);
    EXPECT_TRUE(alpha && beta);
    auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
    EXPECT_TRUE(rdm) << rdm.error().message;
    fb::Buffers::device<double> gamma((n * n) + (n * n * n * n), res.device());
    std::vector<double> root_gammas;
    for (std::size_t r = 0; r < nroots_; ++r) {
      std::vector<double> one_hot(nroots_, 0.0);
      one_hot[r] = 1.0;
      EXPECT_TRUE(rdm->rdm12(res, one_hot, vectors.data(), gamma.data(), gamma.data() + (n * n))
                      .has_value());
      const std::vector<double> root = download(res, gamma.data(), n * n);
      root_gammas.insert(root_gammas.end(), root.begin(), root.end());
    }
    EXPECT_TRUE(
        rdm->rdm12(res, weights, vectors.data(), gamma.data(), gamma.data() + (n * n)).has_value());
    const auto energies = case_["energies"].get<std::vector<double>>();
    return fb::mcscf::Solution{
        std::move(c),
        fb::ci::Roots{energies, std::vector<double>(nroots_, 0.0), std::move(vectors), {}, 0},
        std::inner_product(weights.begin(), weights.end(), energies.begin(), 0.0),
        0.0,
        download(res, gamma.data(), n * n),
        download(res, gamma.data() + (n * n), n * n * n * n),
        std::move(root_gammas),
        {}};
  }

  fb::Result<fb::mcscf::GradientTerms> gradient(const fb::mcscf::Solution &sol,
                                                const std::int32_t root) {
    return fb::mcscf::df_gradient(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(), *b_,
                                  metric_->spherical().data(), sol, ncore_, ncas_, root);
  }

  // Every root's gradient at the case's weights against the fixture's.
  void check(const std::string &name) {
    load(name);
    const fb::mcscf::Solution sol = solution(case_["weights"].get<std::vector<double>>());
    for (std::size_t root = 0; root < nroots_; ++root) {
      const auto g = gradient(sol, static_cast<std::int32_t>(root));
      ASSERT_TRUE(g) << g.error().message;
      const auto want = case_["roots"][root]["gradient"].get<std::vector<double>>();
      ASSERT_EQ(g->total.size(), want.size());
      double worst = 0.0;
      for (std::size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::abs(g->total[i] - want[i]));
      }
      EXPECT_LT(worst, kGradientTol) << name << " root " << root;
      std::cout << std::format("[ saweight ] {} root {}: {:.2e}\n", name, root, worst);
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
  std::vector<double> c_;
  std::vector<double> ci_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t nroots_ = 0;
  std::size_t n_det_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

// Two roots at (0.7, 0.3): the projection is exact for two roots.
TEST_F(McscfGradientWeights, TwoRootsUnequalMatchReference) {
  FBF_REQUIRE_GPU();
  check("h2o_631g_sa2_unequal");
}

TEST_F(McscfGradientWeights, ThreeRootsEqualMatchReference) {
  FBF_REQUIRE_GPU();
  check("h2o_631g_sa3");
}

// Three roots at (0.5, 0.3, 0.2): the rotation between roots 1 and 2 carries
// a multiplier of 0.07 to 0.15, and the gradient is exact only with it.
TEST_F(McscfGradientWeights, ThreeRootsUnequalMatchReference) {
  FBF_REQUIRE_GPU();
  check("h2o_631g_sa3_unequal");
}

} // namespace
