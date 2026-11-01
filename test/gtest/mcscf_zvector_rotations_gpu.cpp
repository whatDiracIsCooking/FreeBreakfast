// GPU cases for the rotations among the averaged roots in mcscf::zvector and
// mcscf::zvectors (#613), against tools/emit/casscf_hessian_reference.py's
// stretched water, CAS(4,4) / 6-31G / cc-pVDZ-JKFIT
// (test/gtest/data/casscf_hessian/): the orbitals and roots loaded from the
// fixture, never re-converged (PySCF's own two-root solution is a saddle
// point). Three roots at (0.5, 0.3, 0.2): each root's Z-vector, z_o, z_c and
// the rotations Theta, by zvector and by all three zvectors methods; and
// lambda^x from mcscf::df_derivatives' SA right-hand sides, its rotation
// rows included, by all three. Two roots at (0.7, 0.3): the rotation carries
// no multiplier. Three at equal weights: none is reported. And the
// refusals: a rotation right-hand side of the wrong length, and Coupled at
// weights that rise with the energies.

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
import free_breakfast.test.ci_oracles;

namespace {

namespace fb = free_breakfast;

// Against the fixture, relative to each block's largest element. The
// fixture's z and lambda^x are exact solves over the same fit; this tree's
// are good to the solves' tolerance (1e-10 on the residual) and, for Schur
// and PerColumn, the CI response's (1e-12 here, tightened from 1e-8).
constexpr double kSolveTol = 1e-9;
// The CI response's tolerance for these cases: Schur's and PerColumn's
// products are good to about it, which would bound z by 1e-8 at the default.
constexpr double kResponseTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

constexpr std::array kSolvers{fb::mcscf::CpSolver::PerColumn, fb::mcscf::CpSolver::Schur,
                              fb::mcscf::CpSolver::Coupled};

std::string_view solver_name(const fb::mcscf::CpSolver solver) {
  switch (solver) {
  case fb::mcscf::CpSolver::PerColumn:
    return "PerColumn";
  case fb::mcscf::CpSolver::Schur:
    return "Schur";
  case fb::mcscf::CpSolver::Coupled:
    return "Coupled";
  }
  return "?";
}

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

// The singlet part of every length-@p dim column of @p v, in place:
// prod_{S = 1..S_max} (S^2 - S(S + 1)) / (0 - S(S + 1)), over the host S^2
// (mcscf_derivative_gpu.cpp's).
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

class McscfZvectorRotations : public ::testing::Test {
protected:
  // One case set up as the DF steps would, the fixture's orbitals made ours
  // (rows over N, in this tree's AO order) and its roots
  // (mcscf_root_rotation_gpu.cpp's); then H~ at them, as df_gradient builds
  // it, with the CI response solved to kResponseTol.
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
    nr_ = weights_.size();
    n_theta_ = nr_ * (nr_ - 1) / 2;
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
    const std::vector<double> c_pyscf = read_array(array("c"), {nao_, nao_});
    std::vector<double> c(nao_ * nao_);
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t k = 0; k < nao_; ++k) {
        c[order[k] + (nao_ * j)] = c_pyscf[k + (nao_ * j)] / norms[order[k]];
      }
    }
    c_.emplace(res, *basis_);
    upload(res, c, c_->data());

    // The CI objects over H_0 at these orbitals, and the roots as ci::Roots:
    // the fixture's vectors, <c_r|H_0|c_r> their energies.
    const auto n = static_cast<std::size_t>(ncas_);
    const auto naux = static_cast<std::int32_t>(naux_);
    auto ints = fb::casscf_integrals(res, *basis_, *molecule_, h_->spherical(), *b_, naux, *c_,
                                     ncore_, ncas_);
    ASSERT_TRUE(ints) << ints.error().message;
    ints_.emplace(std::move(*ints));
    auto alpha = fb::ci::DeviceStringSet::create(res, ncas_, nalpha_);
    auto beta = fb::ci::DeviceStringSet::create(res, ncas_, nbeta_);
    ASSERT_TRUE(alpha && beta);
    alpha_.emplace(std::move(*alpha));
    beta_.emplace(std::move(*beta));
    auto sigma = fb::ci::Sigma::create(res, *alpha_, *beta_);
    auto rdm = fb::ci::Rdm::create(res, *alpha_, *beta_);
    ASSERT_TRUE(sigma && rdm);
    sigma_.emplace(std::move(*sigma));
    rdm_.emplace(std::move(*rdm));
    set_sigma_integrals(res, *ints_, n, *sigma_);
    const std::size_t block = n_det_ * nr_;
    const std::vector<double> ci = read_array(array("ci"), {n_det_, nr_});
    roots_.emplace(fb::ci::Roots{{},
                                 std::vector<double>(nr_, 0.0),
                                 fb::Buffers::device<double>(block, res.device()),
                                 {},
                                 0});
    upload(res, ci, roots_->vectors.data());
    fb::Buffers::device<double> h_c(block, res.device());
    ASSERT_TRUE(
        sigma_->apply(res, static_cast<std::int32_t>(nr_), roots_->vectors.data(), h_c.data())
            .has_value());
    const std::vector<double> hc = download(res, h_c.data(), block);
    const auto want = case_["energies"].get<std::vector<double>>();
    for (std::size_t r = 0; r < nr_; ++r) {
      double e = 0.0;
      for (std::size_t i = 0; i < n_det_; ++i) {
        e += ci[i + (n_det_ * r)] * hc[i + (n_det_ * r)];
      }
      roots_->energies.push_back(e);
      ASSERT_NEAR(e, want[r], 1e-9) << "root " << r;
    }

    fb::Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
    ASSERT_TRUE(
        rdm_->rdm12(res, weights_, roots_->vectors.data(), rdms.data(), rdms.data() + (n * n))
            .has_value());
    auto sa_fock = fb::response::fock(
        res, *molecule_, h_->spherical(), *b_, naux, *c_, ncore_, ncas_, ints_->ppaa, ints_->papa,
        download(res, rdms.data(), n * n), download(res, rdms.data() + (n * n), n * n * n * n));
    ASSERT_TRUE(sa_fock) << sa_fock.error().message;
    sa_fock_.emplace(std::move(*sa_fock));
    auto hessian = fb::response::Hessian::create(res, h_->spherical(), *b_, naux, *c_, ncore_,
                                                 ncas_, ints_->ppaa, ints_->papa, *sa_fock_);
    ASSERT_TRUE(hessian) << hessian.error().message;
    hessian_.emplace(std::move(*hessian));
    fb::mcscf::ResponseIntegrals response_ints{
        nao_,
        static_cast<std::size_t>(ncore_),
        n,
        download(res, sa_fock_->fock_inactive.data(), nao_ * nao_),
        download(res, ints_->ppaa.data(), nao_ * nao_ * n * n),
        download(res, ints_->papa.data(), nao_ * nao_ * n * n)};
    auto relaxed =
        fb::mcscf::RelaxedHessian::create(res, *hessian_, *sigma_, *rdm_, *alpha_, *beta_, *roots_,
                                          weights_, std::move(response_ints), kResponseTol);
    ASSERT_TRUE(relaxed) << relaxed.error().message;
    relaxed_.emplace(std::move(*relaxed));
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

  // @p count nmo x nmo host matrices packed into pairs() x count on the host.
  std::vector<double> pack(const std::vector<double> &full, const std::size_t count) {
    fb::DeviceResources &res = *res_;
    fb::Buffers::device<double> device(full.size(), res.device());
    upload(res, full, device.data());
    fb::Buffers::device<double> packed(pairs() * count, res.device());
    for (std::size_t x = 0; x < count; ++x) {
      EXPECT_TRUE(fb::response::pack_orbital_pairs(res, static_cast<std::int32_t>(nao_), ncore_,
                                                   ncas_, device.data() + (nao_ * nao_ * x),
                                                   packed.data() + (pairs() * x))
                      .has_value());
    }
    return download(res, packed.data(), pairs() * count);
  }

  // A (nroots, nroots, count) Fortran-order Theta (element (s, r)) packed as
  // ZVector::theta: Theta_sr for r < s.
  [[nodiscard]] std::vector<double> pack_theta(const std::vector<double> &full,
                                               const std::size_t count) const {
    std::vector<double> packed;
    for (std::size_t x = 0; x < count; ++x) {
      for (std::size_t r = 0; r < nr_; ++r) {
        for (std::size_t s = r + 1; s < nr_; ++s) {
          packed.push_back(full[s + (nr_ * (r + (nr_ * x)))]);
        }
      }
    }
    return packed;
  }

  // Root I's orbital gradient g_I at the SA orbitals, on the device: the
  // Z-vector's right-hand side, as df_gradient forms it.
  fb::Buffers::device<double> root_gradient(const std::size_t root) {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    std::vector<double> one_hot(nr_, 0.0);
    one_hot[root] = 1.0;
    fb::Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
    EXPECT_TRUE(
        rdm_->rdm12(res, one_hot, roots_->vectors.data(), rdms.data(), rdms.data() + (n * n))
            .has_value());
    auto own = fb::response::fock(res, *molecule_, h_->spherical(), *b_,
                                  static_cast<std::int32_t>(naux_), *c_, ncore_, ncas_, ints_->ppaa,
                                  ints_->papa, download(res, rdms.data(), n * n),
                                  download(res, rdms.data() + (n * n), n * n * n * n));
    EXPECT_TRUE(own) << own.error().message;
    fb::Buffers::device<double> g(pairs(), res.device());
    EXPECT_EQ(wwr::wwrMemcpyAsync(g.data(), own->gradient.data(), sizeof(double) * pairs(),
                                  wwr::wwrMemcpyDeviceToDevice, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(res.sync().has_value());
    return g;
  }

  // One solution's three blocks against the fixture's, each relative to its
  // largest element; the rotations against @p theta. The fixture's roots
  // carry a small non-singlet part and its CI vectors are over the singlet
  // space (data/README.md), so z_c is compared as its singlet part: the raw
  // difference is up to 3e-8 here, at equal weights as much as at unequal.
  void expect_solution(const std::string &what, const std::vector<double> &orbital,
                       std::vector<double> ci, const std::vector<double> &theta,
                       const std::vector<double> &want_orbital, const std::vector<double> &want_ci,
                       const std::vector<double> &want_theta) {
    const auto alpha = fb::ci::StringSet::create(ncas_, nalpha_);
    const auto beta = fb::ci::StringSet::create(ncas_, nbeta_);
    ASSERT_TRUE(alpha && beta);
    project_singlet(*alpha, *beta, ci, n_det_);
    const double d_o = max_diff(orbital, want_orbital);
    const double d_c = max_diff(ci, want_ci);
    const double d_t = max_diff(theta, want_theta);
    EXPECT_LT(d_o, kSolveTol * std::max(1.0, max_abs(want_orbital))) << what << " z_o";
    EXPECT_LT(d_c, kSolveTol * std::max(1.0, max_abs(want_ci))) << what << " z_c";
    EXPECT_LT(d_t, kSolveTol * std::max(1.0, max_abs(want_theta))) << what << " theta";
    std::cout << std::format("[ zvrot    ] {}: z_o {:.2e}, z_c {:.2e}, theta {:.2e} (of {:.3f})\n",
                             what, d_o, d_c, d_t, max_abs(want_theta));
  }

  // Each root's Z-vector against the fixture's z: zvector, then the three
  // zvectors methods on every root's g_I as one block.
  void check_z() {
    fb::DeviceResources &res = *res_;
    const std::size_t block = n_det_ * nr_;
    fb::Buffers::device<double> gradients(pairs() * nr_, res.device());
    std::vector<double> want_o;
    std::vector<double> want_c;
    std::vector<double> want_t;
    for (std::size_t root = 0; root < nr_; ++root) {
      const fb::Buffers::device<double> g = root_gradient(root);
      EXPECT_EQ(wwr::wwrMemcpyAsync(gradients.data() + (pairs() * root), g.data(),
                                    sizeof(double) * pairs(), wwr::wwrMemcpyDeviceToDevice,
                                    res.stream()),
                wwr::wwrSuccess);
      const std::vector<double> o = pack(read_array(root_array(root, "z_orb"), {nao_, nao_}), 1);
      const std::vector<double> c = read_array(root_array(root, "z_ci"), {n_det_, nr_});
      const std::vector<double> t =
          pack_theta(weights_.size() > 1 && !case_["equal_weights"].get<bool>()
                         ? read_array(root_array(root, "z_theta"), {nr_, nr_})
                         : std::vector<double>(nr_ * nr_, 0.0),
                     1);
      auto z = fb::mcscf::zvector(res, *relaxed_, *hessian_, *roots_, g.data());
      ASSERT_TRUE(z) << z.error().message;
      ASSERT_EQ(z->theta.size(), n_theta_);
      expect_solution(std::format("{} root {} zvector", case_["name"].get<std::string>(), root),
                      download(res, z->orbital.data(), pairs()), download(res, z->ci.data(), block),
                      z->theta, o, c, t);
      want_o.insert(want_o.end(), o.begin(), o.end());
      want_c.insert(want_c.end(), c.begin(), c.end());
      want_t.insert(want_t.end(), t.begin(), t.end());
    }
    for (const fb::mcscf::CpSolver solver : kSolvers) {
      auto z =
          fb::mcscf::zvectors(res, *relaxed_, *hessian_, *roots_, static_cast<std::int32_t>(nr_),
                              gradients.data(), nullptr, solver);
      ASSERT_TRUE(z) << solver_name(solver) << ": " << z.error().message;
      ASSERT_EQ(z->theta.size(), n_theta_ * nr_);
      expect_solution(
          std::format("{} zvectors {}", case_["name"].get<std::string>(), solver_name(solver)),
          download(res, z->orbital.data(), pairs() * nr_), download(res, z->ci.data(), block * nr_),
          z->theta, want_o, want_c, want_t);
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
  std::optional<fb::CasscfIntegrals> ints_;
  std::optional<fb::ci::DeviceStringSet> alpha_;
  std::optional<fb::ci::DeviceStringSet> beta_;
  std::optional<fb::ci::Sigma> sigma_;
  std::optional<fb::ci::Rdm> rdm_;
  std::optional<fb::ci::Roots> roots_;
  std::optional<fb::response::Fock> sa_fock_;
  std::optional<fb::response::Hessian> hessian_;
  std::optional<fb::mcscf::RelaxedHessian> relaxed_;
  std::vector<double> weights_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::size_t nr_ = 0;
  std::size_t n_theta_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

// Three roots at (0.5, 0.3, 0.2): Theta_21 carries 0.07 to 0.15, the others
// nothing (casscf_hessian.tex section 5.2).
TEST_F(McscfZvectorRotations, ThreeRootsUnequalZMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3_unequal");
  check_z();
}

// lambda^x over df_derivatives' SA right-hand sides in zvector's convention:
// the CI rows over 2 w_r, the rotation rows over 2 (w_r - w_s), which is
// the derivative Hamiltonian's H~^x_sr.
TEST_F(McscfZvectorRotations, ThreeRootsUnequalLambdaMatchReference) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3_unequal");
  fb::DeviceResources &res = *res_;
  const std::size_t block = n_det_ * nr_;
  auto d = fb::mcscf::df_derivatives(res, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                     *b_, metric_->spherical().data(), *c_, ncore_, ncas_, nalpha_,
                                     nbeta_, roots_->vectors.data(), weights_);
  ASSERT_TRUE(d) << d.error().message;
  const std::vector<double> gradients_host =
      pack(download(res, d->rhs_orbital.data(), nao_ * nao_ * n_x_), n_x_);
  fb::Buffers::device<double> gradients(gradients_host.size(), res.device());
  upload(res, gradients_host, gradients.data());
  std::vector<double> ci_rhs = download(res, d->rhs_ci.data(), block * n_x_);
  for (std::size_t x = 0; x < n_x_; ++x) {
    for (std::size_t r = 0; r < nr_; ++r) {
      for (std::size_t i = 0; i < n_det_; ++i) {
        ci_rhs[i + (n_det_ * (r + (nr_ * x)))] /= 2.0 * weights_[r];
      }
    }
  }
  fb::Buffers::device<double> ci_rhs_device(ci_rhs.size(), res.device());
  upload(res, ci_rhs, ci_rhs_device.data());
  std::vector<double> theta_rhs;
  for (std::size_t x = 0; x < n_x_; ++x) {
    for (std::size_t r = 0; r < nr_; ++r) {
      for (std::size_t s = r + 1; s < nr_; ++s) {
        // dE/dTheta_sr over 2 (w_r - w_s): H~^x_sr itself.
        const std::size_t sr = s + (nr_ * (r + (nr_ * x)));
        const double t = d->rhs_theta[sr] / (2.0 * (weights_[r] - weights_[s]));
        EXPECT_NEAR(t, d->hamiltonian[sr], 1e-12);
        theta_rhs.push_back(t);
      }
    }
  }

  const std::vector<double> want_o =
      pack(read_array(array("lambda_orb"), {nao_, nao_, n_x_}), n_x_);
  const std::vector<double> want_c = read_array(array("lambda_ci"), {n_det_, nr_, n_x_});
  const std::vector<double> want_t =
      pack_theta(read_array(array("lambda_theta"), {nr_, nr_, n_x_}), n_x_);
  for (const fb::mcscf::CpSolver solver : kSolvers) {
    auto z =
        fb::mcscf::zvectors(res, *relaxed_, *hessian_, *roots_, static_cast<std::int32_t>(n_x_),
                            gradients.data(), ci_rhs_device.data(), solver, 1e-10, 200, theta_rhs);
    ASSERT_TRUE(z) << solver_name(solver) << ": " << z.error().message;
    expect_solution(std::format("lambda^x {}", solver_name(solver)),
                    download(res, z->orbital.data(), pairs() * n_x_),
                    download(res, z->ci.data(), block * n_x_), z->theta, want_o, want_c, want_t);
  }
  // Without the rotation rows lambda^x is another one.
  auto z = fb::mcscf::zvectors(res, *relaxed_, *hessian_, *roots_, static_cast<std::int32_t>(n_x_),
                               gradients.data(), ci_rhs_device.data());
  ASSERT_TRUE(z) << z.error().message;
  EXPECT_GT(max_diff(z->theta, want_t), 1e-3);
}

// Two roots at (0.7, 0.3): the one rotation is a parameter, but it carries
// no multiplier (the fixture's is 5e-13; the design observes it below 3e-7).
TEST_F(McscfZvectorRotations, TwoRootsUnequalCarryNoRotation) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2_unequal");
  check_z();
}

// At equal weights the rotations are redundant: none is reported, and z is
// the fixture's.
TEST_F(McscfZvectorRotations, ThreeRootsEqualReportNone) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3");
  check_z();
}

TEST_F(McscfZvectorRotations, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3_unequal");
  fb::DeviceResources &res = *res_;
  const fb::Buffers::device<double> g = root_gradient(0);
  // A rotation right-hand side of the wrong length.
  const std::vector<double> short_t(n_theta_ - 1, 0.0);
  auto one = fb::mcscf::zvector(res, *relaxed_, *hessian_, *roots_, g.data(), nullptr, 1e-10, 200,
                                short_t);
  ASSERT_FALSE(one);
  EXPECT_EQ(one.error().kind, fb::ErrorKind::InvalidConfig) << one.error().message;
  auto many = fb::mcscf::zvectors(res, *relaxed_, *hessian_, *roots_, 1, g.data(), nullptr,
                                  fb::mcscf::CpSolver::Schur, 1e-10, 200, short_t);
  ASSERT_FALSE(many);
  EXPECT_EQ(many.error().kind, fb::ErrorKind::InvalidConfig) << many.error().message;

  // Coupled at weights that rise with the energies: the rotations' diagonal
  // 2 (w_r - w_s)(E_s - E_r) is negative there, which its conjugate
  // gradients cannot take.
  const std::vector<double> rising{0.2, 0.3, 0.5};
  const auto n = static_cast<std::size_t>(ncas_);
  fb::mcscf::ResponseIntegrals response_ints{
      nao_,
      static_cast<std::size_t>(ncore_),
      n,
      download(res, sa_fock_->fock_inactive.data(), nao_ * nao_),
      download(res, ints_->ppaa.data(), nao_ * nao_ * n * n),
      download(res, ints_->papa.data(), nao_ * nao_ * n * n)};
  auto over = fb::mcscf::RelaxedHessian::create(res, *hessian_, *sigma_, *rdm_, *alpha_, *beta_,
                                                *roots_, rising, std::move(response_ints));
  ASSERT_TRUE(over) << over.error().message;
  auto coupled = fb::mcscf::zvectors(res, *over, *hessian_, *roots_, 1, g.data(), nullptr,
                                     fb::mcscf::CpSolver::Coupled);
  ASSERT_FALSE(coupled);
  EXPECT_EQ(coupled.error().kind, fb::ErrorKind::InvalidConfig) << coupled.error().message;
  EXPECT_NE(coupled.error().message.find("rise"), std::string::npos) << coupled.error().message;
}

} // namespace
