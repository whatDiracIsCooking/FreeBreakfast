// GPU cases for mcscf::root_rotation, the root-rotation term of an
// equal-weight state-averaged DF-CASSCF Hessian (casscf_hessian.tex eqs.
// theta and rot), against tools/emit/casscf_hessian_reference.py's fixtures
// (test/gtest/data/casscf_hessian/): the orbitals and roots loaded from the
// fixture, never re-converged (PySCF's own two-root solution is a saddle
// point). Two roots and three at equal weights: each root's couplings and
// eq. rot, first over the fixture's lambda^x, then over lambda^x solved here
// by mcscf::zvectors from mcscf::df_derivatives' right-hand sides (the chain
// a Hessian runs). Beside them the refusals: a near-degenerate pair (the
// fixture's roots given energies a small gap apart), unequal weights and bad
// sizes.

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
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;

namespace {

namespace fb = free_breakfast;

// Against the fixture over its own lambda^x, absolute: a tenth of the issue's
// 1e-8. The couplings are H~^x (df_derivatives', good to 1e-13 here) plus a
// contraction of the fixture's lambda^x, the same formula over the same fit.
constexpr double kFixtureTol = 1e-9;
// Over lambda^x solved by zvectors: its products are good to the CI
// response's tolerance (1e-8 of their right-hand side), which bounds lambda^x
// and so the term; the issue's 1e-8.
constexpr double kSolvedTol = 1e-8;

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

// The active-space Hamiltonian of ints in the row-major layout the sigma
// takes (mcscf_zvectors_gpu.cpp's).
void set_sigma_integrals(fb::DeviceResources &res, const fb::CasscfIntegrals &ints,
                         const std::size_t n, fb::ci::Sigma &sigma) {
  const std::vector<double> h_col = download(res, ints.hamiltonian.h_eff.data(), n * n);
  const std::vector<double> eri_col = download(res, ints.hamiltonian.eri.data(), n * n * n * n);
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }
  EXPECT_TRUE(sigma.set_integrals(res, ints.hamiltonian.e_core, h_eff, eri).has_value());
}

class McscfRootRotation : public ::testing::Test {
protected:
  // One case set up as the DF steps would: the bases placed, the core
  // Hamiltonian, the fit B and its L, the fixture's orbitals made ours (rows
  // over N) and its roots (mcscf_derivative_gpu.cpp's).
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
  [[nodiscard]] std::size_t pairs() const {
    const auto core = static_cast<std::size_t>(ncore_);
    const auto act = static_cast<std::size_t>(ncas_);
    return (core * (nao_ - core)) + (act * (nao_ - core - act));
  }
  [[nodiscard]] std::vector<double> energies() const {
    return case_["energies"].get<std::vector<double>>();
  }

  fb::Result<fb::mcscf::Derivatives> derivatives() {
    return fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                     *b_, metric_->spherical().data(), *c_, ncore_, ncas_, nalpha_,
                                     nbeta_, roots_->data(), weights_);
  }

  fb::Result<fb::mcscf::RootRotation> run(const fb::mcscf::Derivatives &d, const double *kappa,
                                          std::span<const double> energies,
                                          std::span<const double> weights,
                                          const double max_rotation = 10.0) {
    return fb::mcscf::root_rotation(*res_, *molecule_, *basis_, h_->spherical(), *b_, *c_, ncore_,
                                    ncas_, nalpha_, nbeta_, roots_->data(), weights, energies, d,
                                    kappa, max_rotation);
  }

  // The nmo x nmo x n device stack @p full packed, slab by slab, into
  // pairs() x n.
  fb::Buffers::device<double> pack(const double *full) {
    fb::Buffers::device<double> packed(pairs() * n_x_, res_->device());
    for (std::size_t x = 0; x < n_x_; ++x) {
      EXPECT_TRUE(fb::response::pack_orbital_pairs(*res_, static_cast<std::int32_t>(nao_), ncore_,
                                                   ncas_, full + (nao_ * nao_ * x),
                                                   packed.data() + (pairs() * x))
                      .has_value());
    }
    EXPECT_TRUE(res_->sync().has_value());
    return packed;
  }

  // The fixture's lambda^x orbital part, packed.
  fb::Buffers::device<double> fixture_kappa() {
    const std::vector<double> full = read_array(array("lambda_orb"), {nao_, nao_, n_x_});
    fb::Buffers::device<double> device(full.size(), res_->device());
    upload(*res_, full, device.data());
    return pack(device.data());
  }

  // lambda^x's orbital part solved here: zvectors over H~ at the fixture's
  // orbitals and roots, on d's SA right-hand sides, the CI rows taken to its
  // convention (2 w_r Q H~^x c_r over 2 w_r). Against the fixture's on the way.
  fb::Buffers::device<double> solved_kappa(const fb::mcscf::Derivatives &d) {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    const std::size_t nr = weights_.size();
    const auto naux = static_cast<std::int32_t>(naux_);
    auto ints = fb::casscf_integrals(res, *basis_, *molecule_, h_->spherical(), *b_, naux, *c_,
                                     ncore_, ncas_);
    EXPECT_TRUE(ints) << ints.error().message;
    auto alpha = fb::ci::DeviceStringSet::create(res, ncas_, nalpha_);
    auto beta = fb::ci::DeviceStringSet::create(res, ncas_, nbeta_);
    EXPECT_TRUE(alpha && beta);
    auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
    auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
    EXPECT_TRUE(sigma && rdm);
    set_sigma_integrals(res, *ints, n, *sigma);
    const std::size_t block = n_det_ * nr;

    // The roots as ci::Roots: the fixture's vectors, <c_r|H_0|c_r> their
    // energies.
    fb::ci::Roots roots{
        {}, std::vector<double>(nr, 0.0), fb::Buffers::device<double>(block, res.device()), {}, 0};
    EXPECT_EQ(wwr::wwrMemcpyAsync(roots.vectors.data(), roots_->data(), sizeof(double) * block,
                                  wwr::wwrMemcpyDeviceToDevice, res.stream()),
              wwr::wwrSuccess);
    fb::Buffers::device<double> h_c(block, res.device());
    EXPECT_TRUE(sigma->apply(res, static_cast<std::int32_t>(nr), roots.vectors.data(), h_c.data())
                    .has_value());
    const std::vector<double> c_host = download(res, roots.vectors.data(), block);
    const std::vector<double> hc_host = download(res, h_c.data(), block);
    const std::vector<double> want = energies();
    for (std::size_t r = 0; r < nr; ++r) {
      double e = 0.0;
      for (std::size_t i = 0; i < n_det_; ++i) {
        e += c_host[i + (n_det_ * r)] * hc_host[i + (n_det_ * r)];
      }
      roots.energies.push_back(e);
      EXPECT_NEAR(e, want[r], 1e-9) << "root " << r;
    }

    fb::Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
    EXPECT_TRUE(rdm->rdm12(res, weights_, roots.vectors.data(), rdms.data(), rdms.data() + (n * n))
                    .has_value());
    const std::vector<double> gamma = download(res, rdms.data(), n * n);
    const std::vector<double> gamma2 = download(res, rdms.data() + (n * n), n * n * n * n);
    auto sa_fock = fb::response::fock(res, *molecule_, h_->spherical(), *b_, naux, *c_, ncore_,
                                      ncas_, ints->ppaa, ints->papa, gamma, gamma2);
    EXPECT_TRUE(sa_fock) << sa_fock.error().message;
    auto hessian = fb::response::Hessian::create(res, h_->spherical(), *b_, naux, *c_, ncore_,
                                                 ncas_, ints->ppaa, ints->papa, *sa_fock);
    EXPECT_TRUE(hessian) << hessian.error().message;
    const fb::mcscf::ResponseIntegrals response_ints{
        nao_,
        static_cast<std::size_t>(ncore_),
        n,
        download(res, sa_fock->fock_inactive.data(), nao_ * nao_),
        download(res, ints->ppaa.data(), nao_ * nao_ * n * n),
        download(res, ints->papa.data(), nao_ * nao_ * n * n)};
    auto relaxed = fb::mcscf::RelaxedHessian::create(res, *hessian, *sigma, *rdm, *alpha, *beta,
                                                     roots, weights_, response_ints);
    EXPECT_TRUE(relaxed) << relaxed.error().message;

    const fb::Buffers::device<double> gradients = pack(d.rhs_orbital.data());
    std::vector<double> ci_rhs = download(res, d.rhs_ci.data(), block * n_x_);
    for (std::size_t x = 0; x < n_x_; ++x) {
      for (std::size_t r = 0; r < nr; ++r) {
        for (std::size_t i = 0; i < n_det_; ++i) {
          ci_rhs[i + (n_det_ * (r + (nr * x)))] /= 2.0 * weights_[r];
        }
      }
    }
    fb::Buffers::device<double> ci_rhs_device(ci_rhs.size(), res.device());
    upload(res, ci_rhs, ci_rhs_device.data());
    auto z = fb::mcscf::zvectors(res, *relaxed, *hessian, roots, static_cast<std::int32_t>(n_x_),
                                 gradients.data(), ci_rhs_device.data());
    EXPECT_TRUE(z) << z.error().message;

    const std::vector<double> mine = download(res, z->orbital.data(), pairs() * n_x_);
    const fb::Buffers::device<double> fixture = fixture_kappa();
    const std::vector<double> want_kappa = download(res, fixture.data(), pairs() * n_x_);
    EXPECT_LT(max_diff(mine, want_kappa), kSolvedTol * std::max(1.0, max_abs(want_kappa)))
        << "lambda^x's orbital part, max " << max_abs(want_kappa);
    return std::move(z->orbital);
  }

  // Each root's couplings, rotations and eq. rot against the fixture's.
  void check(const fb::mcscf::RootRotation &rot, const double tol) {
    const std::size_t nr = weights_.size();
    const std::vector<double> e = energies();
    ASSERT_EQ(rot.coupling.size(), nr * nr * n_x_);
    ASSERT_EQ(rot.theta.size(), nr * nr * n_x_);
    ASSERT_EQ(rot.hessian.size(), n_x_ * n_x_ * nr);
    for (std::size_t i = 0; i < nr; ++i) {
      // coupling is (nroots, n), row I zero: element (s, x) at s + nroots x.
      const std::vector<double> coupling = read_array(root_array(i, "coupling"), {nr, n_x_});
      double worst_coupling = 0.0;
      double worst_theta = 0.0;
      for (std::size_t s = 0; s < nr; ++s) {
        for (std::size_t x = 0; x < n_x_; ++x) {
          const std::size_t slab = nr * nr * x;
          const double want = coupling[s + (nr * x)];
          worst_coupling =
              std::max(worst_coupling, std::abs(rot.coupling[i + (nr * s) + slab] - want));
          // theta^x_sI = H^x_Is / (E_I - E_s), zero on the diagonal.
          const double theta = s == i ? 0.0 : want / (e[i] - e[s]);
          worst_theta = std::max(worst_theta, std::abs(rot.theta[s + (nr * i) + slab] - theta));
          EXPECT_EQ(rot.theta[s + (nr * i) + slab], -rot.theta[i + (nr * s) + slab]);
        }
      }
      EXPECT_LT(worst_coupling, tol) << "root " << i << " couplings, max " << max_abs(coupling);
      // theta is the coupling over a gap of at least 0.065 hartree here.
      EXPECT_LT(worst_theta, 20.0 * tol) << "root " << i << " theta";

      const std::vector<double> want = read_array(root_array(i, "rot"), {n_x_, n_x_});
      const std::vector<double> mine(
          rot.hessian.begin() + static_cast<std::ptrdiff_t>(n_x_ * n_x_ * i),
          rot.hessian.begin() + static_cast<std::ptrdiff_t>(n_x_ * n_x_ * (i + 1)));
      EXPECT_LT(max_diff(mine, want), tol) << "root " << i << " eq. rot, max " << max_abs(want);
      // The factor 2: one copy of the product would miss by half the term.
      std::vector<double> half = want;
      for (double &v : half) {
        v *= 0.5;
      }
      EXPECT_GT(max_diff(mine, half), 1e-3) << "root " << i;
    }
  }

  void check_solved(const std::string &name);

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

TEST_F(McscfRootRotation, TwoRootsMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2");
  auto d = derivatives();
  ASSERT_TRUE(d) << d.error().message;
  const fb::Buffers::device<double> kappa = fixture_kappa();
  auto rot = run(*d, kappa.data(), energies(), weights_);
  ASSERT_TRUE(rot) << rot.error().message;
  check(*rot, kFixtureTol);
}

// Three equal roots, where the term is as large as the Hessian itself.
TEST_F(McscfRootRotation, ThreeRootsMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3");
  auto d = derivatives();
  ASSERT_TRUE(d) << d.error().message;
  const fb::Buffers::device<double> kappa = fixture_kappa();
  auto rot = run(*d, kappa.data(), energies(), weights_);
  ASSERT_TRUE(rot) << rot.error().message;
  check(*rot, kFixtureTol);
}

// The chain a Hessian runs: df_derivatives, zvectors on its right-hand sides,
// then the term over that lambda^x.
void McscfRootRotation::check_solved(const std::string &name) {
  load(name);
  auto d = derivatives();
  ASSERT_TRUE(d) << d.error().message;
  const fb::Buffers::device<double> kappa = solved_kappa(*d);
  auto rot = run(*d, kappa.data(), energies(), weights_);
  ASSERT_TRUE(rot) << rot.error().message;
  check(*rot, kSolvedTol);
}

TEST_F(McscfRootRotation, TwoRootsOverSolvedLambdaMatchReference) {
  FBF_REQUIRE_GPU();
  check_solved("h2o_631g_sa2");
}

TEST_F(McscfRootRotation, ThreeRootsOverSolvedLambdaMatchReference) {
  FBF_REQUIRE_GPU();
  check_solved("h2o_631g_sa3");
}

TEST_F(McscfRootRotation, OneRootGivesZero) {
  FBF_REQUIRE_GPU();
  load("h2o_631g");
  auto d = derivatives();
  ASSERT_TRUE(d) << d.error().message;
  const fb::Buffers::device<double> kappa = fixture_kappa();
  auto rot = run(*d, kappa.data(), energies(), weights_);
  ASSERT_TRUE(rot) << rot.error().message;
  EXPECT_EQ(max_abs(rot->coupling), 0.0);
  EXPECT_EQ(max_abs(rot->theta), 0.0);
  EXPECT_EQ(max_abs(rot->hessian), 0.0);
  EXPECT_EQ(rot->hessian.size(), n_x_ * n_x_);
}

// A near-degenerate pair, constructed: the two-root case's roots given
// energies 1e-3 hartree apart, against couplings up to 0.055, so
// |H^x| / gap is about 55 per bohr, past the default 10. The message names
// the gap. A coincident pair is refused whatever the threshold.
TEST_F(McscfRootRotation, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2");
  auto d = derivatives();
  ASSERT_TRUE(d) << d.error().message;
  const fb::Buffers::device<double> kappa = fixture_kappa();
  const std::vector<double> e = energies();
  const auto expect_error = [](const auto &r, const fb::ErrorKind kind, std::string_view what,
                               std::string_view text = "mcscf::root_rotation") {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, kind) << what;
    EXPECT_NE(r.error().message.find(text), std::string::npos) << what << ": " << r.error().message;
  };

  const std::vector<double> near{e[0], e[0] + 1e-3};
  expect_error(run(*d, kappa.data(), near, weights_), fb::ErrorKind::Numerical, "near-degenerate",
               "1.000e-03 hartree apart");
  // The threshold, not the gap alone, refuses it.
  EXPECT_TRUE(run(*d, kappa.data(), near, weights_, 1e3));
  const std::vector<double> same{e[0], e[0]};
  expect_error(run(*d, kappa.data(), same, weights_, 1e300), fb::ErrorKind::Numerical, "coincident",
               "0.000e+00 hartree apart");

  const std::vector<double> unequal{0.7, 0.3};
  expect_error(run(*d, kappa.data(), e, unequal), fb::ErrorKind::InvalidConfig, "unequal weights");
  const std::vector<double> one_energy{e[0]};
  expect_error(run(*d, kappa.data(), one_energy, weights_), fb::ErrorKind::InvalidConfig,
               "energies' count");
  expect_error(run(*d, kappa.data(), e, weights_, 0.0), fb::ErrorKind::InvalidConfig,
               "max_rotation 0");
  const std::vector<double> three{1.0 / 3, 1.0 / 3, 1.0 / 3};
  const std::vector<double> three_e{e[0], e[1], e[1] + 1.0};
  expect_error(run(*d, kappa.data(), three_e, three), fb::ErrorKind::InvalidConfig,
               "derivatives of two roots");
}

} // namespace
