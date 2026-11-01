// GPU cases for the z_o part of a state-averaged DF-CASSCF Hessian's L_xy:
// E_xy (casscf_hessian.tex eq. exy) differentiated along the orbital rotation
// C -> C (1 + e K_z) at fixed MO-basis RDMs, by hessian::df_casscf_exy and
// df_casscf_two_electron with a rotation and mcscf::df_derivatives'
// fock_rotated, at the averaged RDMs and K_z = the fixture's z_orb. The
// fixture's L_xy z part (data/casscf_hessian/, z_one_electron ...
// exy_z) is that at z's densities, orbital and CI parts together, so the
// test adds the z_c part the merged entry points already give (E_xy at the
// roots' transition RDMs with their first-order change, by the +/-
// polarization of rdm12 and df_derivatives) and compares each piece and the
// sum. The orbitals, roots and z are loaded from the fixture, never
// re-converged: PySCF's two-root solutions are saddle points. Two roots at
// equal weights and at (0.7, 0.3), three at equal weights and at
// (0.5, 0.3, 0.2), every root; the z_o part's symmetry and translational sum
// rule; and the refusal of a rotation without every orbital.

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
// the total of L_xy's z part.
constexpr double kHessianTol = 1e-8;
// Each z_o piece's asymmetry and the z_o part's sum over atoms, absolute.
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

// One state-averaged case set up as the DF steps would: the bases placed,
// h, the fit B and its L, the fixture's orbitals made ours (rows over N, in
// this tree's AO order), its roots on the device, and the averaged RDMs.
class HessianDfCasscfRotation : public ::testing::Test {
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
    weights_ = case_["weights"].get<std::vector<double>>();
    nroots_ = weights_.size();
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

    const std::vector<double> ci =
        read_array(case_["arrays"]["ci"].get<std::string>(), {n_det_, nroots_});
    ASSERT_FALSE(HasFailure());
    ci_host_ = ci;
    roots_.emplace(upload(*res_, ci));
    auto alpha = fb::ci::DeviceStringSet::create(*res_, ncas_, nalpha_);
    ASSERT_TRUE(alpha) << alpha.error().message;
    auto beta = fb::ci::DeviceStringSet::create(*res_, ncas_, nbeta_);
    ASSERT_TRUE(beta) << beta.error().message;
    alpha_.emplace(std::move(*alpha));
    beta_.emplace(std::move(*beta));
    auto rdm = fb::ci::Rdm::create(*res_, *alpha_, *beta_);
    ASSERT_TRUE(rdm) << rdm.error().message;
    ASSERT_EQ(static_cast<std::size_t>(rdm->size()), n_det_);
    rdm_.emplace(std::move(*rdm));
    std::tie(gamma_, gamma2_) = rdms(*roots_, weights_);
  }

  // gamma and Gamma of the columns of @p vectors at @p weights, row-major.
  std::pair<std::vector<double>, std::vector<double>>
  rdms(const fb::Buffers::device<double> &vectors, const std::vector<double> &weights) {
    const auto n = static_cast<std::size_t>(ncas_);
    fb::Buffers::device<double> gamma(n * n, res_->device());
    fb::Buffers::device<double> gamma2(n * n * n * n, res_->device());
    EXPECT_TRUE(rdm_->rdm12(*res_, weights, vectors.data(), gamma.data(), gamma2.data()));
    return {download(*res_, gamma.data(), gamma.num_elements()),
            download(*res_, gamma2.data(), gamma2.num_elements())};
  }

  [[nodiscard]] std::vector<double> root_array(const std::size_t root, const std::string &key,
                                               const std::vector<std::size_t> &shape) const {
    return read_array(case_["roots"][root]["arrays"][key].get<std::string>(), shape);
  }

  fb::Result<fb::mcscf::Derivatives> derivatives(const fb::Buffers::device<double> &vectors,
                                                 const double *rotation) {
    return fb::mcscf::df_derivatives(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                     *b_, metric_->spherical().data(), *c_matrix_, ncore_, ncas_,
                                     nalpha_, nbeta_, vectors.data(), weights_, rotation);
  }

  fb::Result<fb::hessian::CasscfExy> exy(std::span<const double> gamma,
                                         std::span<const double> gamma2,
                                         const fb::Buffers::device<double> &fock_x,
                                         const double *rotation) {
    return fb::hessian::df_casscf_exy(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                      *b_, metric_->spherical().data(), *c_, ncore_, ncas_, gamma,
                                      gamma2, fock_x, rotation);
  }

  // Root I's L_xy z part, orbital and CI, by piece against the fixture.
  void check_root(const std::size_t root) {
    const std::size_t nn = nao_ * nao_;
    const std::size_t n2 = n_x_ * n_x_;
    const std::vector<std::size_t> square{n_x_, n_x_};
    const std::vector<double> kz = root_array(root, "z_orb", {nao_, nao_});
    const std::vector<double> z_ci = root_array(root, "z_ci", {n_det_, nroots_});
    std::vector<double> z_theta(nroots_ * nroots_, 0.0);
    if (case_["roots"][root]["arrays"].contains("z_theta")) {
      z_theta = root_array(root, "z_theta", {nroots_, nroots_});
    }
    ASSERT_FALSE(HasFailure());

    // z_o: F^[x] and E_xy at the K_z-rotated averaged densities.
    const auto start = std::chrono::steady_clock::now();
    const fb::Buffers::device<double> k = upload(*res_, kz);
    auto rotated = derivatives(*roots_, k.data());
    ASSERT_TRUE(rotated) << rotated.error().message;
    ASSERT_TRUE(rotated->fock_rotated.has_value());
    ASSERT_EQ(rotated->fock_rotated->num_elements(), nn * n_x_);
    auto z_o = exy(gamma_, gamma2_, *rotated->fock_rotated, k.data());
    ASSERT_TRUE(z_o) << z_o.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // z_c: the roots' change dc_r = z_ci[:, r] + Sum_s Theta_sr c_s, and
    // Sum_r w_r (T(dc_r, c_r) + T(c_r, dc_r)) as half the difference of the
    // RDMs of c_r + dc_r and c_r - dc_r at the weights (their core parts
    // cancel); E_xy is linear in (gamma, Gamma, F^[x]) at a fixed core, so its
    // z_c part is E_xy at the averaged RDMs plus those, less E_xy at the
    // averaged RDMs alone.
    std::vector<double> plus(ci_host_);
    std::vector<double> minus(ci_host_);
    for (std::size_t r = 0; r < nroots_; ++r) {
      for (std::size_t i = 0; i < n_det_; ++i) {
        double dc = z_ci[i + (n_det_ * r)];
        for (std::size_t s = 0; s < nroots_; ++s) {
          dc += z_theta[s + (nroots_ * r)] * ci_host_[i + (n_det_ * s)];
        }
        plus[i + (n_det_ * r)] += dc;
        minus[i + (n_det_ * r)] -= dc;
      }
    }
    const fb::Buffers::device<double> plus_d = upload(*res_, plus);
    const fb::Buffers::device<double> minus_d = upload(*res_, minus);
    auto [g_plus, g2_plus] = rdms(plus_d, weights_);
    auto [g_minus, g2_minus] = rdms(minus_d, weights_);
    auto f_plus = derivatives(plus_d, nullptr);
    ASSERT_TRUE(f_plus) << f_plus.error().message;
    auto f_minus = derivatives(minus_d, nullptr);
    ASSERT_TRUE(f_minus) << f_minus.error().message;
    std::vector<double> fock_t = download(*res_, f_plus->fock.data(), nn * n_x_);
    const std::vector<double> fock_minus = download(*res_, f_minus->fock.data(), nn * n_x_);
    for (std::size_t i = 0; i < fock_t.size(); ++i) {
      fock_t[i] = 0.5 * (fock_t[i] - fock_minus[i]);
    }
    std::vector<double> gamma_t(gamma_);
    std::vector<double> gamma2_t(gamma2_);
    for (std::size_t i = 0; i < gamma_t.size(); ++i) {
      gamma_t[i] += 0.5 * (g_plus[i] - g_minus[i]);
    }
    for (std::size_t i = 0; i < gamma2_t.size(); ++i) {
      gamma2_t[i] += 0.5 * (g2_plus[i] - g2_minus[i]);
    }
    const fb::Buffers::device<double> fock_t_d = upload(*res_, fock_t);
    const fb::Buffers::device<double> zero_fock =
        upload(*res_, std::vector<double>(nn * n_x_, 0.0));
    auto with_t = exy(gamma_t, gamma2_t, fock_t_d, nullptr);
    ASSERT_TRUE(with_t) << with_t.error().message;
    auto without_t = exy(gamma_, gamma2_, zero_fock, nullptr);
    ASSERT_TRUE(without_t) << without_t.error().message;

    std::string report;
    for (const auto &[key, orbital, with, without] :
         {std::tuple{"z_one_electron", &z_o->one_electron, &with_t->one_electron,
                     &without_t->one_electron},
          std::tuple{"z_two_electron", &z_o->two_electron.total, &with_t->two_electron.total,
                     &without_t->two_electron.total},
          std::tuple{"z_connection_second", &z_o->connection_second, &with_t->connection_second,
                     &without_t->connection_second},
          std::tuple{"z_connection_fock", &z_o->connection_fock, &with_t->connection_fock,
                     &without_t->connection_fock},
          std::tuple{"z_connection_twice", &z_o->connection_twice, &with_t->connection_twice,
                     &without_t->connection_twice},
          std::tuple{"exy_z", &z_o->exy, &with_t->exy, &without_t->exy}}) {
      const std::vector<double> ref = root_array(root, key, square);
      ASSERT_EQ(orbital->size(), n2) << key;
      std::vector<double> total(n2);
      for (std::size_t i = 0; i < n2; ++i) {
        total[i] = (*orbital)[i] + ((*with)[i] - (*without)[i]);
      }
      const double worst = max_diff(total, ref);
      const double asym = asymmetry(*orbital, n_x_);
      EXPECT_LE(worst, kHessianTol)
          << key << " root " << root << ": max |z - ref| " << worst << " of " << max_abs(ref);
      EXPECT_LE(asym, kSymmetryTol) << key << " root " << root;
      report += std::format("; {} max |z| {:.3e} (z_o {:.3e}), |z - ref| {:.2e}, z_o asym {:.1e}",
                            key, max_abs(ref), max_abs(*orbital), worst, asym);
    }
    EXPECT_EQ(max_abs(z_o->nuclear_repulsion), 0.0);
    const double drift = sum_rule(z_o->exy, n_x_);
    EXPECT_LE(drift, kSumRuleTol) << "root " << root;
    std::cout << std::format("[ z_o      ] {} root {}: {} coordinates, z_o part {:.2f} s, sum "
                             "rule {:.2e}{}\n",
                             case_["name"].get<std::string>(), root, n_x_, elapsed, drift, report);
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_FALSE(HasFailure());
    for (std::size_t root = 0; root < nroots_; ++root) {
      check_root(root);
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
  std::optional<fb::Buffers::device<double>> c_;
  std::optional<fb::SphericalMatrixAO> c_matrix_;
  std::optional<fb::Buffers::device<double>> roots_;
  std::optional<fb::ci::DeviceStringSet> alpha_;
  std::optional<fb::ci::DeviceStringSet> beta_;
  std::optional<fb::ci::Rdm> rdm_;
  std::vector<double> c_host_;
  std::vector<double> ci_host_;
  std::vector<double> weights_;
  std::vector<double> gamma_;
  std::vector<double> gamma2_;
  std::string aux_stamp_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t n_det_ = 0;
  std::size_t nroots_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfCasscfRotation, TwoRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa2");
}

TEST_F(HessianDfCasscfRotation, TwoRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa2_unequal");
}

TEST_F(HessianDfCasscfRotation, ThreeRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa3");
}

TEST_F(HessianDfCasscfRotation, ThreeRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa3_unequal");
}

// A rotation reaches every orbital, so C must hold them all.
TEST_F(HessianDfCasscfRotation, RotationNeedsEveryOrbital) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa2");
  ASSERT_FALSE(HasFatalFailure());
  const fb::Buffers::device<double> k = upload(*res_, std::vector<double>(nao_ * nao_, 0.0));
  const auto nocc = static_cast<std::size_t>(ncore_ + ncas_);
  const fb::Buffers::device<double> c_occ((nao_ * nocc) + 1, res_->device());
  const auto r = fb::hessian::df_casscf_two_electron(*res_, *basis_, classes_, *aux_, *b_,
                                                     metric_->spherical().data(), c_occ, ncore_,
                                                     ncas_, gamma_, gamma2_, k.data());
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(r.error().message.find("hessian::df_casscf_two_electron"), std::string::npos);
}

} // namespace
