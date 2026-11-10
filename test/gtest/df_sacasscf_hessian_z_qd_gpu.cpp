// GPU cases for hessian::df_sacasscf_z and mcscf::ThirdDerivative at a
// QD-NEVPT2 Lagrangian's multipliers (Z, {z_r}, Theta), state K = 1 of
// nevpt2::qd_sc_lagrangian, instead of one root's z (docs/nevpt2/nevpt2.tex
// Eq. qd-Lxx's third derivative, its CI curvature the root set's, Eq.
// qd-curv). On data/qd_nevpt2/'s h2co_skew_ccpvdz_sa3 (C1, three roots at
// equal weights) and h2co_skew_ccpvdz_sa3_weighted (0.4, 0.4, 0.2: Theta on
// the pairs with root 2, and every pair's rotation a parameter), from this
// tree's own DF-SA-CASSCF through the registry.
//
// Each term against a Richardson ladder of
//
//     F(R, lambda) = Z . g_SA + Sum_r z_r . g_ci,r (+ Theta . g_Theta)
//                  = d/de E_SA(R; lambda + e z) at e = 0,
//
// evaluated on the host from this tree's integrals at each displaced geometry
// along the symmetric connection C(R) = C (C^T S(R) C)^-1/2 and the root
// set's exponential parametrization C(R) e^K, e^A c_r (A = Sum_r (d_r c_r^T
// - c_r d_r^T) + Sum_{r<s} Theta_sr (c_s c_r^T - c_r c_s^T)), the
// exponentials' derivatives exact (their series), only the RDMs from
// ci::Rdm. Along two nuclear coordinates A and B and their sum e, the
// parameters moving along x_e (df_sacasscf_cp's lambda^x with, at the
// weighted case, df_qd_nevpt2_tangents' rotations, Eq. qd-theta's for the
// equal pair):
//
//     T(x_e, x_e, z)                          against d2/dt2 F(R_0, t x_e)
//     L_ee + 2 L_e,lambda . x_e + T(x_e, x_e, z) against d2/dt2 F(R_0 + t e, t x_e)
//
// L_ee the z part of L_xy, L_e,lambda df_sacasscf_z's covector (no
// right-hand side: the multipliers' terms alone), T ThirdDerivative's. The
// sum's ladder with the two coordinates' gives the mixed element. Visibility,
// each > 10x the bar on some element: the root-set curvature's share, the
// third term's ladder in the root set's parametrization less its ladder with
// each root on its own sphere (Part I's -(u . v) c curvature); and at
// unequal weights Theta's share, the ladders at z less at z with Theta zeroed.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.response.zvector;
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar, absolute, for each term against its ladder; a share is
// visible above ten times it.
constexpr double kLadderTol = 1e-7;
constexpr double kVisible = 10.0 * kLadderTol;
// The ladders' step, bohr along a coordinate and t along x_e (per bohr),
// with half of it beside it (Richardson).
constexpr double kStep = 4e-3;
// The host energy at the reference against the roots' average.
constexpr double kEnergyTol = 1e-9;
// Terms of the exponentials' series: the arguments' norms stay below 0.2.
constexpr std::int32_t kSeriesTerms = 30;
// The reference SA-CASSCF's ||g|| and |Delta E| (ThirdDerivative reads the
// roots as eigenvectors; df_qd_nevpt2_tangents_gpu.cpp's convergence).
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;
// The QD state, K >= 1, and the two coordinates (O's z, the first H's y).
constexpr std::int32_t kState = 1;
constexpr std::size_t kCoordA = 5;
constexpr std::size_t kCoordB = 7;

std::vector<double> download(fb::DeviceResources &res, const double *device, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *device) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(device, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

double max_abs(std::span<const double> v) {
  return std::ranges::fold_left(
      v, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// C (m x n) = op(A) B, column-major; op(A) is A (m x k), or A^T for A k x m.
std::vector<double> host_gemm(std::size_t m, std::size_t k, std::size_t n,
                              const std::vector<double> &a, bool transpose_a,
                              const std::vector<double> &b) {
  std::vector<double> c(m * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t l = 0; l < k; ++l) {
      const double blj = b[l + (k * j)];
      for (std::size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += (transpose_a ? a[l + (k * i)] : a[i + (m * l)]) * blj;
      }
    }
  }
  return c;
}

// A^-1/2 of a symmetric A near the identity (||1 - A|| < 1), by the coupled
// Newton-Schulz iteration: Y -> A^1/2, Z -> A^-1/2.
std::vector<double> inverse_sqrt(const std::vector<double> &a, std::size_t n) {
  std::vector<double> y = a;
  std::vector<double> z(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z[i * (n + 1)] = 1.0;
  }
  for (std::int32_t it = 0; it < 20; ++it) {
    std::vector<double> t = host_gemm(n, n, n, z, false, y);
    double off = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      off = std::max(off, std::abs(t[i] - (i % (n + 1) == 0 ? 1.0 : 0.0)));
      t[i] = -0.5 * t[i];
    }
    if (off < 1e-14) {
      break;
    }
    for (std::size_t i = 0; i < n; ++i) {
      t[i * (n + 1)] += 1.5;
    }
    y = host_gemm(n, n, n, y, false, t);
    z = host_gemm(n, n, n, t, false, z);
  }
  return z;
}

// X = L^-1 B for the lower triangle of the n x n @p l and the n x cols @p b.
std::vector<double> lower_solve(const std::vector<double> &l, std::size_t n, std::vector<double> b,
                                std::size_t cols) {
  for (std::size_t j = 0; j < cols; ++j) {
    double *x = b.data() + (n * j);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < i; ++k) {
        x[i] -= l[i + (n * k)] * x[k];
      }
      x[i] /= l[i * (n + 1)];
    }
  }
  return b;
}

// exp(X) of a small n x n X by its series.
std::vector<double> expm(const std::vector<double> &x, std::size_t n) {
  std::vector<double> out(n * n, 0.0);
  std::vector<double> term(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out[i * (n + 1)] = 1.0;
    term[i * (n + 1)] = 1.0;
  }
  for (std::int32_t k = 1; k < kSeriesTerms; ++k) {
    term = host_gemm(n, n, n, term, false, x);
    for (std::size_t i = 0; i < n * n; ++i) {
      term[i] /= k;
      out[i] += term[i];
    }
  }
  return out;
}

// e^-X d/de e^(X + e Y) at e = 0: Sum_k (-1)^k / (k + 1)! ad_X^k Y.
std::vector<double> dexp(const std::vector<double> &x, const std::vector<double> &y,
                         std::size_t n) {
  std::vector<double> out = y;
  std::vector<double> term = y;
  for (std::int32_t k = 1; k < kSeriesTerms; ++k) {
    const std::vector<double> xt = host_gemm(n, n, n, x, false, term);
    const std::vector<double> tx = host_gemm(n, n, n, term, false, x);
    for (std::size_t i = 0; i < n * n; ++i) {
      term[i] = -(xt[i] - tx[i]) / (k + 1);
      out[i] += term[i];
    }
  }
  return out;
}

double richardson(const double coarse, const double fine) {
  return ((4 * fine) - coarse) / 3;
}

// How the roots move with the CI part of the parameters: the root set's one
// exponential (ThirdDerivative's, Eq. qd-K), or each root's own, Part I's
// single-state sphere, which differs from it only at second order.
enum class Roots : std::uint8_t { Joint, Separate };

// One case: the registry's SA-CASSCF, its CP responses, the QD Lagrangian's
// multipliers, the host's AO data at any geometry and F at any (h_MO, L) and
// parameters.
class HessianDfSaCasscfZQd : public ::testing::Test {
protected:
  // Everything at one geometry, on the host: S, h, the metric M and the
  // three-center T (unfitted).
  struct Geometry {
    std::vector<double> s, h, m, t;
  };
  using Integrals = std::pair<std::vector<double>, std::vector<double>>;

  [[nodiscard]] fb::DeviceResources &res() const { return node_->device(0); }
  [[nodiscard]] const fb::Molecule &molecule() const {
    return calc_->get<fb::MoleculeOutput>()->molecule;
  }
  [[nodiscard]] const fb::BasisSet &basis() const {
    return calc_->get<fb::BasisSetOutput>()->basis;
  }
  [[nodiscard]] std::span<const fb::PrimitivePairClass> classes() const {
    return calc_->get<fb::PrimitivePairClassesOutput>()->classes;
  }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc_->get<fb::EriOutput>()->eri);
  }
  [[nodiscard]] const fb::SphericalMatrixAO &h() const {
    return calc_->get<fb::CoreHamiltonianOutput>()->h;
  }

  Geometry at(const std::vector<fb::Atom> &where) {
    fb::DeviceResources &r = res();
    Geometry g;
    const fb::Molecule shifted(r, where);
    auto placed = fb::place(stamp_->primitive_normalized(), shifted);
    auto aux = fb::place(aux_stamp_->primitive_normalized(), shifted);
    EXPECT_TRUE(placed && aux);
    if (!placed || !aux) {
      return g;
    }
    EXPECT_EQ(placed->nao_sphe(), nmo_);
    EXPECT_EQ(aux->nao_sphe(), naux_);
    auto pairs = fb::primitive_pair_classes(r, *placed);
    auto basis_u = fb::upload_spherical_transform_matrix(r, *placed);
    auto aux_u = fb::upload_spherical_transform_matrix(r, *aux);
    EXPECT_TRUE(pairs && basis_u && aux_u);
    if (!pairs || !basis_u || !aux_u) {
      return g;
    }
    const std::size_t nn = nmo_ * nmo_;
    fb::MatrixAOPair s_pair(r, *placed, *basis_u);
    EXPECT_TRUE(fb::overlap(r, *pairs, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(r));
    fb::MatrixAOPair h_pair(r, *placed, *basis_u);
    EXPECT_TRUE(fb::core_hamiltonian(r, *pairs, shifted, h_pair.cartesian()));
    EXPECT_TRUE(h_pair.cartesian_to_spherical(r));
    fb::MatrixAOPair metric(r, *aux, std::move(*aux_u));
    EXPECT_TRUE(fb::metric(r, *aux, metric));
    fb::Buffers::device<double> t(nn * naux_, r.device());
    EXPECT_TRUE(fb::three_center(r, *placed, *pairs, *aux, t.data()));
    g.s = download(r, s_pair.spherical().data(), nn);
    g.h = download(r, h_pair.spherical().data(), nn);
    g.m = download(r, metric.spherical().data(), naux_ * naux_);
    g.t = download(r, t.data(), nn * naux_);
    return g;
  }

  // The case through the registry, then lambda^x, the multipliers and the
  // analytic terms.
  void load(const std::string &name) {
    const auto doc =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / "qd_nevpt2.json");
    ASSERT_TRUE(doc.has_value()) << doc.error().message;
    fb::json entry;
    for (const fb::json &e : doc->at("cases")) {
      if (e.at("name").get<std::string>() == name) {
        entry = e;
      }
    }
    ASSERT_FALSE(entry.is_null()) << "no case " << name;
    name_ = name;
    weights_ = entry.at("weights").get<std::vector<double>>();
    nroots_ = weights_.size();
    const std::vector<bool> rotates = fb::response::rotating_pairs(weights_);
    rotations_ = std::ranges::contains(rotates, true);
    node_ = std::make_unique<fb::NodeResources>(
        fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces});
    const fb::test::ScratchDir dir("df_sacasscf_hessian_z_qd");
    std::string xyz = std::format("{}\n{}\n", entry.at("atoms").size(), name);
    for (const fb::json &atom : entry.at("atoms")) {
      const auto coord = atom.at(1).get<std::vector<double>>();
      xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                         coord[0] * fb::kAngstromPerBohr<double>,
                         coord[1] * fb::kAngstromPerBohr<double>,
                         coord[2] * fb::kAngstromPerBohr<double>);
    }
    ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
    std::string weights;
    for (const double w : weights_) {
      weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
    }
    const std::string toml = std::format(
        "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
        "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
        "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol = {:g}\n"
        "conv_tol_grad = {:g}\n",
        entry.at("basis").get<std::string>(), entry.at("aux_basis").at("name").get<std::string>(),
        entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(), nroots_,
        weights, kConvTol, kConvTolGrad);
    ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
    auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    const fb::Status ran = calc->run(*node_);
    ASSERT_TRUE(ran.has_value()) << ran.error().message;
    calc_.emplace(std::move(*calc));
    fb::DeviceResources &r = res();
    const auto &cas = calc_->get<fb::CasscfOutput>();
    ASSERT_TRUE(cas.has_value());
    ASSERT_LT(cas->gradient_norm, kConvTolGrad);
    ncore_ = cas->ncore;
    ncas_ = cas->ncas;
    nocc_ = static_cast<std::size_t>(ncore_ + ncas_);
    const std::int32_t two_s = molecule().multiplicity() - 1;
    nbeta_ = (cas->nelecas - two_s) / 2;
    nalpha_ = cas->nelecas - nbeta_;
    nmo_ = basis().nao_sphe();
    naux_ = fit().aux.nao_sphe();
    n_x_ = 3 * basis().natoms();
    const auto n2 = static_cast<std::size_t>(ncas_) * static_cast<std::size_t>(ncas_);
    std::vector<double> gamma(n2, 0.0);
    for (std::size_t root = 0; root < nroots_; ++root) {
      for (std::size_t i = 0; i < n2; ++i) {
        gamma[i] += cas->weights[root] * cas->rdm1[(root * n2) + i];
      }
    }
    solution_.emplace(fb::mcscf::Solution{
        .c = fb::SphericalMatrixAO(r, basis()),
        .roots = {.energies = cas->energies,
                  .spin_squares = cas->spin_squares,
                  .vectors = fb::Buffers::device<double>(cas->vectors.size(), r.device()),
                  .davidson = {},
                  .fp32_iterations = 0},
        .energy = cas->energy,
        .gradient_norm = cas->gradient_norm,
        .gamma = std::move(gamma),
        .gamma2 = cas->rdm2,
        .root_gammas = cas->rdm1,
        .iterations = cas->iterations,
        .weights = cas->weights});
    upload(r, cas->vectors, solution_->roots.vectors.data());
    ASSERT_EQ(wwr::wwrMemcpyAsync(solution_->c.data(), cas->c.data(),
                                  cas->c.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(r.sync().has_value());
    c_host_ = download(r, solution_->c.data(), nmo_ * nmo_);
    roots_host_ = cas->vectors;
    dim_ = roots_host_.size() / nroots_;
    for (std::size_t root = 0; root < nroots_; ++root) {
      sa_energy_ += weights_[root] * cas->energies[root];
    }
    auto alpha = fb::ci::DeviceStringSet::create(r, ncas_, nalpha_);
    auto beta = fb::ci::DeviceStringSet::create(r, ncas_, nbeta_);
    ASSERT_TRUE(alpha && beta);
    alpha_.emplace(std::move(*alpha));
    beta_.emplace(std::move(*beta));
    auto rdm = fb::ci::Rdm::create(r, *alpha_, *beta_);
    ASSERT_TRUE(rdm) << rdm.error().message;
    rdm_.emplace(std::move(*rdm));
    ASSERT_EQ(static_cast<std::size_t>(rdm_->size()), dim_);
    auto stamp = fb::resolve_basis(entry.at("basis").get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    stamp_.emplace(std::move(*stamp));
    auto aux_stamp = fb::resolve_aux_basis(entry.at("aux_basis").at("name").get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    aux_stamp_.emplace(std::move(*aux_stamp));
    atoms_ = molecule().atoms();
    for (std::size_t q = 0; q < nocc_; ++q) {
      for (std::size_t p = q < static_cast<std::size_t>(ncore_) ? ncore_ : nocc_; p < nmo_; ++p) {
        pairs_.emplace_back(p, q);
      }
    }
    n_theta_ = rotations_ ? nroots_ * (nroots_ - 1) / 2 : 0;
    size_ = pairs_.size() + (dim_ * nroots_) + n_theta_;

    // lambda^x, and at unequal weights every pair's rotation per coordinate
    // (Eq. qd-theta's for an equal pair, the CP solve's for the rest).
    const auto t0 = std::chrono::steady_clock::now();
    auto cp =
        fb::hessian::df_sacasscf_cp(r, molecule(), basis(), classes(), fit().aux, h(), fit().b,
                                    fit().l.data(), *solution_, weights_, ncore_, ncas_);
    ASSERT_TRUE(cp.has_value()) << cp.error().message;
    std::vector<double> theta;
    if (rotations_) {
      const auto none = [](std::int32_t, std::int32_t, const double *, const double *,
                           const double *, const double *) -> fb::Status { return {}; };
      auto tangents = fb::hessian::df_qd_nevpt2_tangents(
          r, molecule(), basis(), classes(), fit().aux, h(), fit().b, fit().l.data(), *solution_,
          weights_, ncore_, ncas_, cp->cp, false, none);
      ASSERT_TRUE(tangents.has_value()) << tangents.error().message;
      theta = std::move(tangents->theta);
      ASSERT_EQ(theta.size(), n_theta_ * n_x_);
    }
    const std::size_t n_ci = dim_ * nroots_;
    const std::vector<double> orbital = download(r, cp->cp.orbital.data(), pairs_.size() * n_x_);
    const std::vector<double> ci = download(r, cp->cp.ci.data(), n_ci * n_x_);
    const auto column = [&](const std::size_t x) {
      std::vector<double> v;
      v.insert(v.end(), orbital.begin() + static_cast<std::ptrdiff_t>(x * pairs_.size()),
               orbital.begin() + static_cast<std::ptrdiff_t>((x + 1) * pairs_.size()));
      v.insert(v.end(), ci.begin() + static_cast<std::ptrdiff_t>(x * n_ci),
               ci.begin() + static_cast<std::ptrdiff_t>((x + 1) * n_ci));
      v.insert(v.end(), theta.begin() + static_cast<std::ptrdiff_t>(x * n_theta_),
               theta.begin() + static_cast<std::ptrdiff_t>((x + 1) * n_theta_));
      return off_roots(std::move(v));
    };
    x_a_ = column(kCoordA);
    x_b_ = column(kCoordB);

    // The QD Lagrangian's multipliers for state K, in ThirdDerivative's
    // layout.
    auto lagrangian = fb::nevpt2::qd_sc_lagrangian(r, molecule(), basis(), fit().aux, h(), fit().b,
                                                   *solution_, ncore_, ncas_, kState, 0);
    ASSERT_TRUE(lagrangian.has_value()) << lagrangian.error().message;
    const fb::response::ZVector &m = lagrangian->multipliers;
    z_ = download(r, m.orbital.data(), pairs_.size());
    const std::vector<double> z_ci = download(r, m.ci.data(), n_ci);
    z_.insert(z_.end(), z_ci.begin(), z_ci.end());
    ASSERT_EQ(m.theta.size(), nroots_ * (nroots_ - 1) / 2);
    if (rotations_) {
      z_.insert(z_.end(), m.theta.begin(), m.theta.end());
    }
    z_ = off_roots(std::move(z_));
    ASSERT_EQ(z_.size(), size_);
    std::cout << std::format(
        "[ SA z QD  ] {}: E {}, ||g|| {:.1e}; CP and multipliers {:.2f} s; max |Z| {:.2e}, "
        "|z_c| {:.2e}, |Theta| {:.2e}; max |x_A| {:.2e}, |x_B| {:.2e}\n",
        name, cas->energies, cas->gradient_norm,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
        max_abs(std::span(z_).first(pairs_.size())),
        max_abs(std::span(z_).subspan(pairs_.size(), n_ci)),
        max_abs(std::span(z_).subspan(pairs_.size() + n_ci)), max_abs(x_a_), max_abs(x_b_));
    ref_ = at(atoms_);
    ASSERT_FALSE(HasFailure());
  }

  // @p v with each root's CI part taken off every root, as the operators
  // take it.
  [[nodiscard]] std::vector<double> off_roots(std::vector<double> v) const {
    double *d = v.data() + pairs_.size();
    for (std::size_t r = 0; r < nroots_; ++r) {
      for (std::size_t s = 0; s < nroots_; ++s) {
        const double *c = roots_host_.data() + (s * dim_);
        const double along = std::inner_product(c, c + dim_, d + (r * dim_), 0.0);
        for (std::size_t i = 0; i < dim_; ++i) {
          d[(r * dim_) + i] -= along * c[i];
        }
      }
    }
    return v;
  }

  // The analytic terms at @p z: df_sacasscf_z's two (no right-hand side) and
  // ThirdDerivative's T over the columns x_A, x_B, (i, j) at i + 2 j.
  struct Analytic {
    std::vector<double> lxy;
    std::vector<double> lxlambda; // size x n_x on the host
    std::array<double, 4> third{};
  };
  Analytic analytic(const std::vector<double> &z_host) {
    fb::DeviceResources &r = res();
    Analytic a;
    fb::Buffers::device<double> z(size_, r.device());
    upload(r, z_host, z.data());
    auto terms = fb::hessian::df_sacasscf_z(
        r, molecule(), basis(), classes(), fit().aux, h(), fit().b, fit().l.data(), solution_->c,
        ncore_, ncas_, nalpha_, nbeta_, solution_->roots.vectors, weights_, {}, z);
    EXPECT_TRUE(terms.has_value()) << terms.error().message;
    if (!terms.has_value()) {
      return a;
    }
    a.lxy = terms->lxy.exy;
    a.lxlambda = download(r, terms->lxlambda.data(), size_ * n_x_);
    auto op = fb::mcscf::ThirdDerivative::create(
        r, h(), fit().b, static_cast<std::int32_t>(naux_), solution_->c, ncore_, ncas_, *alpha_,
        *beta_, solution_->roots.vectors.data(), weights_, z.data());
    EXPECT_TRUE(op.has_value()) << op.error().message;
    if (!op.has_value()) {
      return a;
    }
    EXPECT_EQ(static_cast<std::size_t>(op->size()), size_);
    std::vector<double> lambda = x_a_;
    lambda.insert(lambda.end(), x_b_.begin(), x_b_.end());
    fb::Buffers::device<double> l(lambda.size(), r.device());
    upload(r, lambda, l.data());
    fb::Buffers::device<double> y(lambda.size(), r.device());
    EXPECT_TRUE(op->apply(r, 2, l.data(), y.data()).has_value());
    const std::vector<double> y_host = download(r, y.data(), lambda.size());
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t i = 0; i < 2; ++i) {
        a.third[i + (2 * j)] =
            std::inner_product(lambda.begin() + static_cast<std::ptrdiff_t>(i * size_),
                               lambda.begin() + static_cast<std::ptrdiff_t>((i + 1) * size_),
                               y_host.begin() + static_cast<std::ptrdiff_t>(j * size_), 0.0);
      }
    }
    return a;
  }

  // (h_MO, L) at a geometry: h_MO = X^T h X and L_Q = Sum_P X^T T_P X
  // (L_M^-T)_PQ, X = C (C^T S C)^-1/2 the connection and L_M M's Cholesky
  // factor there, L_Q at p + nmo (q + nmo Q).
  [[nodiscard]] Integrals integrals(const Geometry &g) const {
    const std::size_t nn = nmo_ * nmo_;
    const std::vector<double> sc = host_gemm(nmo_, nmo_, nmo_, g.s, false, c_host_);
    const std::vector<double> x =
        host_gemm(nmo_, nmo_, nmo_, c_host_, false,
                  inverse_sqrt(host_gemm(nmo_, nmo_, nmo_, c_host_, true, sc), nmo_));
    const auto to_mo = [&](const std::vector<double> &ao) {
      return host_gemm(nmo_, nmo_, nmo_, x, true, host_gemm(nmo_, nmo_, nmo_, ao, false, x));
    };
    std::vector<double> h_mo = to_mo(g.h);
    std::vector<double> kt(naux_ * nn);
    std::vector<double> tp(nn);
    for (std::size_t p = 0; p < naux_; ++p) {
      for (std::size_t mn = 0; mn < nn; ++mn) {
        tp[mn] = g.t[p + (naux_ * mn)];
      }
      const std::vector<double> mo = to_mo(tp);
      for (std::size_t pq = 0; pq < nn; ++pq) {
        kt[p + (naux_ * pq)] = mo[pq];
      }
    }
    std::vector<double> chol = g.m;
    for (std::size_t j = 0; j < naux_; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t i = j; i < naux_; ++i) {
          chol[i + (naux_ * j)] -= chol[i + (naux_ * k)] * chol[j + (naux_ * k)];
        }
      }
      const double pivot = std::sqrt(chol[j * (naux_ + 1)]);
      for (std::size_t i = j; i < naux_; ++i) {
        chol[i + (naux_ * j)] /= pivot;
      }
    }
    const std::vector<double> lt = lower_solve(chol, naux_, std::move(kt), nn);
    std::vector<double> l(nn * naux_);
    for (std::size_t pq = 0; pq < nn; ++pq) {
      for (std::size_t q = 0; q < naux_; ++q) {
        l[pq + (nn * q)] = lt[q + (naux_ * pq)];
      }
    }
    return {std::move(h_mo), std::move(l)};
  }

  // K (nmo x nmo, antisymmetric) of a vector's packed pairs.
  [[nodiscard]] std::vector<double> kappa(const std::vector<double> &v) const {
    std::vector<double> k(nmo_ * nmo_, 0.0);
    for (std::size_t i = 0; i < pairs_.size(); ++i) {
      const auto [p, q] = pairs_[i];
      k[p + (nmo_ * q)] = v[i];
      k[q + (nmo_ * p)] = -v[i];
    }
    return k;
  }

  // The roots' generator of a vector (dim x dim): Sum_r (d_r c_r^T - c_r
  // d_r^T) over every root, or over @p only alone, plus the rotations Sum_{r<s}
  // Theta_sr (c_s c_r^T - c_r c_s^T).
  [[nodiscard]] std::vector<double> generator(const std::vector<double> &v,
                                              std::optional<std::size_t> only) const {
    std::vector<double> a(dim_ * dim_, 0.0);
    const auto add = [&](const double *u, const double *w, const double scale) {
      for (std::size_t j = 0; j < dim_; ++j) {
        for (std::size_t i = 0; i < dim_; ++i) {
          a[i + (dim_ * j)] += scale * ((u[i] * w[j]) - (w[i] * u[j]));
        }
      }
    };
    for (std::size_t r = 0; r < nroots_; ++r) {
      if (!only || *only == r) {
        add(v.data() + pairs_.size() + (r * dim_), roots_host_.data() + (r * dim_), 1.0);
      }
    }
    const double *theta = v.data() + pairs_.size() + (dim_ * nroots_);
    std::size_t k = 0;
    for (std::size_t r = 0; r < nroots_ && n_theta_ != 0; ++r) {
      for (std::size_t s = r + 1; s < nroots_; ++s, ++k) {
        add(roots_host_.data() + (s * dim_), roots_host_.data() + (r * dim_), theta[k]);
      }
    }
    return a;
  }

  // gamma (ncas^2) and Gamma (ncas^4), row-major, of the nroots vectors @p y
  // at the weights.
  std::pair<std::vector<double>, std::vector<double>> rdms(const std::vector<double> &y) {
    fb::DeviceResources &r = res();
    const auto n = static_cast<std::size_t>(ncas_);
    fb::Buffers::device<double> v(y.size(), r.device());
    upload(r, y, v.data());
    fb::Buffers::device<double> out((n * n) + (n * n * n * n), r.device());
    EXPECT_TRUE(rdm_->rdm12(r, weights_, v.data(), out.data(), out.data() + (n * n)).has_value());
    const std::vector<double> all = download(r, out.data(), out.num_elements());
    return {std::vector<double>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n * n)),
            std::vector<double>(all.begin() + static_cast<std::ptrdiff_t>(n * n), all.end())};
  }

  // F at the integrals @p ints and the parameters @p lambda: d/de of E_SA at
  // C e^(K(lambda) + e K(z)) and the roots moved by lambda + e z (@p mode),
  // exactly (the exponentials' derivatives by their series), the orbital
  // part by dE/dU_o at the averaged densities and the CI part as
  // Sum_r w_r 2 c_r'^T H dc_r' = Sum_r w_r (E(c_r' + dc_r') - E(c_r' - dc_r')) / 2
  // (the densities differenced first). With @p energy, E_SA at lambda.
  double zg(const Integrals &ints, const std::vector<double> &lambda, const std::vector<double> &z,
            const Roots mode, const bool energy = false) {
    const auto &[h_mo, l_mo] = ints;
    const std::size_t nmo = nmo_;
    const std::size_t nocc = nocc_;
    const std::size_t no2 = nocc * nocc;
    const auto core = static_cast<std::size_t>(ncore_);
    const auto n = static_cast<std::size_t>(ncas_);
    const std::vector<double> k = kappa(lambda);
    const std::vector<double> u = expm(k, nmo);
    const std::vector<double> du = host_gemm(nmo, nmo, nmo, u, false, dexp(k, kappa(z), nmo));
    std::vector<double> c1(dim_ * nroots_);
    std::vector<double> dc(dim_ * nroots_);
    const auto move = [&](const std::optional<std::size_t> only, const std::size_t first,
                          const std::size_t count) {
      const std::vector<double> a = generator(lambda, only);
      const std::vector<double> y = expm(a, dim_);
      const std::vector<double> c(roots_host_.begin() + static_cast<std::ptrdiff_t>(first * dim_),
                                  roots_host_.begin() +
                                      static_cast<std::ptrdiff_t>((first + count) * dim_));
      const std::vector<double> moved = host_gemm(dim_, dim_, count, y, false, c);
      const std::vector<double> dmoved =
          host_gemm(dim_, dim_, count, y, false,
                    host_gemm(dim_, dim_, count, dexp(a, generator(z, only), dim_), false, c));
      std::ranges::copy(moved, c1.begin() + static_cast<std::ptrdiff_t>(first * dim_));
      std::ranges::copy(dmoved, dc.begin() + static_cast<std::ptrdiff_t>(first * dim_));
    };
    if (mode == Roots::Joint) {
      move(std::nullopt, 0, nroots_);
    } else {
      for (std::size_t r = 0; r < nroots_; ++r) {
        move(r, r, 1);
      }
    }

    // U_o, h U_o, h_o = U_o^T h U_o; per Q, L_Q U_o and M_Q = U_o^T L_Q U_o.
    const std::vector<double> uo(u.begin(), u.begin() + static_cast<std::ptrdiff_t>(nmo * nocc));
    const std::vector<double> huo = host_gemm(nmo, nmo, nocc, h_mo, false, uo);
    const std::vector<double> ho = host_gemm(nocc, nmo, nocc, uo, true, huo);
    std::vector<std::vector<double>> luo(naux_);
    std::vector<std::vector<double>> mq(naux_);
    for (std::size_t q = 0; q < naux_; ++q) {
      const std::vector<double> lq(l_mo.begin() + static_cast<std::ptrdiff_t>(q * nmo * nmo),
                                   l_mo.begin() + static_cast<std::ptrdiff_t>((q + 1) * nmo * nmo));
      luo[q] = host_gemm(nmo, nmo, nocc, lq, false, uo);
      mq[q] = host_gemm(nocc, nmo, nocc, uo, true, luo[q]);
    }
    std::vector<double> g(no2 * no2, 0.0);
    for (std::size_t q = 0; q < naux_; ++q) {
      for (std::size_t rs = 0; rs < no2; ++rs) {
        const double m_rs = mq[q][rs];
        for (std::size_t pq = 0; pq < no2; ++pq) {
          g[pq + (no2 * rs)] += mq[q][pq] * m_rs;
        }
      }
    }
    // D and Gamma over the occupied orbitals of the vectors y at the weights
    // (the core weighted by Sum_r w_r |y_r|^2).
    const auto full = [&](const std::vector<double> &y) {
      const auto [g1, g2] = rdms(y);
      double n0 = 0.0;
      for (std::size_t r = 0; r < nroots_; ++r) {
        const double *yr = y.data() + (r * dim_);
        n0 += weights_[r] * std::inner_product(yr, yr + dim_, yr, 0.0);
      }
      std::vector<double> d(no2, 0.0);
      std::vector<double> gg(no2 * no2, 0.0);
      const auto at = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double & {
        return gg[p + (nocc * q) + (no2 * (r + (nocc * s)))];
      };
      for (std::size_t i = 0; i < core; ++i) {
        d[i * (nocc + 1)] = 2 * n0;
        for (std::size_t j = 0; j < core; ++j) {
          at(i, i, j, j) += 4 * n0;
          at(i, j, j, i) -= 2 * n0;
        }
        for (std::size_t t = 0; t < n; ++t) {
          for (std::size_t w = 0; w < n; ++w) {
            const double gtw = g1[(t * n) + w];
            at(i, i, core + t, core + w) += 2 * gtw;
            at(core + t, core + w, i, i) += 2 * gtw;
            at(core + t, i, i, core + w) -= gtw;
            at(i, core + w, core + t, i) -= gtw;
          }
        }
      }
      for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t w = 0; w < n; ++w) {
          d[(core + t) + (nocc * (core + w))] = g1[(t * n) + w];
          for (std::size_t v = 0; v < n; ++v) {
            for (std::size_t x2 = 0; x2 < n; ++x2) {
              at(core + t, core + w, core + v, core + x2) +=
                  g2[(((((t * n) + w) * n) + v) * n) + x2];
            }
          }
        }
      }
      return std::pair{std::move(d), std::move(gg)};
    };
    const auto [d, gg] = full(c1);
    if (energy) {
      double e = 0.0;
      for (std::size_t i = 0; i < no2; ++i) {
        e += ho[i] * d[i];
      }
      for (std::size_t i = 0; i < no2 * no2; ++i) {
        e += 0.5 * g[i] * gg[i];
      }
      return e;
    }

    // The orbital part: dE/dU_o = h U_o (D + D^T) + Sum_Q L_Q U_o (W_Q + W_Q^T),
    // W_Q,pq = Sum_rs Gs_pqrs M_Q,rs, against dU_o.
    std::vector<double> dsym(no2);
    for (std::size_t p = 0; p < nocc; ++p) {
      for (std::size_t q = 0; q < nocc; ++q) {
        dsym[p + (nocc * q)] = d[p + (nocc * q)] + d[q + (nocc * p)];
      }
    }
    std::vector<double> grad = host_gemm(nmo, nocc, nocc, huo, false, dsym);
    const std::size_t mo_occ = nmo * nocc;
    std::vector<double> lu_stack(mo_occ * naux_);
    std::vector<double> m_stack(naux_ * no2);
    for (std::size_t q = 0; q < naux_; ++q) {
      std::ranges::copy(luo[q], lu_stack.begin() + static_cast<std::ptrdiff_t>(q * mo_occ));
      for (std::size_t rs = 0; rs < no2; ++rs) {
        m_stack[q + (naux_ * rs)] = mq[q][rs];
      }
    }
    const std::vector<double> t = host_gemm(mo_occ, naux_, no2, lu_stack, false, m_stack);
    const auto gs = [&](std::size_t p, std::size_t q, std::size_t rs) {
      const std::size_t pq = p + (nocc * q);
      return 0.5 * (gg[pq + (no2 * rs)] + gg[rs + (no2 * pq)]);
    };
    for (std::size_t p = 0; p < nocc; ++p) {
      for (std::size_t q = 0; q < nocc; ++q) {
        for (std::size_t rs = 0; rs < no2; ++rs) {
          const double weight = gs(p, q, rs) + gs(q, p, rs);
          const double *tq = t.data() + (nmo * q) + (mo_occ * rs);
          for (std::size_t m = 0; m < nmo; ++m) {
            grad[m + (nmo * p)] += tq[m] * weight;
          }
        }
      }
    }
    double orbital = 0.0;
    for (std::size_t i = 0; i < nmo * nocc; ++i) {
      orbital += du[i] * grad[i];
    }
    // The CI part from the difference of the densities of c' + dc' and
    // c' - dc', taken before the contraction.
    std::vector<double> plus = c1;
    std::vector<double> minus = c1;
    for (std::size_t i = 0; i < plus.size(); ++i) {
      plus[i] += dc[i];
      minus[i] -= dc[i];
    }
    const auto [d_plus, g_plus] = full(plus);
    const auto [d_minus, g_minus] = full(minus);
    double ci = 0.0;
    for (std::size_t i = 0; i < no2; ++i) {
      ci += ho[i] * (d_plus[i] - d_minus[i]);
    }
    for (std::size_t i = 0; i < no2 * no2; ++i) {
      ci += 0.5 * g[i] * (g_plus[i] - g_minus[i]);
    }
    return orbital + (0.5 * ci);
  }

  // The atoms with coordinate x moved by @p step.
  [[nodiscard]] std::vector<fb::Atom> moved(std::vector<fb::Atom> atoms, std::size_t x,
                                            double step) const {
    fb::Vec<double, 3> &r = atoms[x / 3].position;
    (x % 3 == 0 ? r.x : x % 3 == 1 ? r.y : r.z) += step;
    return atoms;
  }

  // Every term at z against its ladder along A, B and A + B, and the shares.
  void check() {
    const std::size_t n = n_x_;
    const auto begin = std::chrono::steady_clock::now();
    const Analytic got = analytic(z_);
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
    ASSERT_FALSE(HasFailure());
    ASSERT_EQ(got.lxy.size(), n * n);

    // The host energy is the roots' average at the reference.
    const Integrals ref_ints = integrals(ref_);
    const std::vector<double> zero(size_, 0.0);
    EXPECT_NEAR(zg(ref_ints, zero, z_, Roots::Joint, true),
                sa_energy_ - fb::nuclear_repulsion_energy(molecule()), kEnergyTol);

    // The three directions: A, B and A + B, with their responses.
    struct Direction {
      std::string name;
      std::vector<std::pair<std::size_t, double>> coordinates;
      std::vector<double> x;
    };
    std::vector<double> x_sum = x_a_;
    for (std::size_t i = 0; i < size_; ++i) {
      x_sum[i] += x_b_[i];
    }
    const std::array<Direction, 3> directions{
        Direction{"A", {{kCoordA, 1.0}}, x_a_}, Direction{"B", {{kCoordB, 1.0}}, x_b_},
        Direction{"A + B", {{kCoordA, 1.0}, {kCoordB, 1.0}}, x_sum}};
    // T(x_e, x_e, z), L_ee and 2 L_e,lambda . x_e of an analytic result.
    const auto third_of = [](const Analytic &a, const std::size_t e) {
      const auto &t = a.third;
      return e == 0 ? t[0] : e == 1 ? t[3] : t[0] + t[1] + t[2] + t[3];
    };
    const auto total_of = [&](const Analytic &a, const std::size_t e) {
      const Direction &dir = directions[e];
      double lxy = 0.0;
      double lxl = 0.0;
      for (const auto &[x, wx] : dir.coordinates) {
        for (const auto &[y, wy] : dir.coordinates) {
          lxy += wx * wy * a.lxy[x + (n * y)];
        }
        for (std::size_t i = 0; i < size_; ++i) {
          lxl += wx * a.lxlambda[i + (size_ * x)] * dir.x[i];
        }
      }
      return lxy + (2.0 * lxl) + third_of(a, e);
    };

    const auto ladder_begin = std::chrono::steady_clock::now();
    // d2/dt2 of F along a direction, from F at -t, 0 and t, Richardson.
    const auto second = [&](const auto &f) {
      const double f0 = f(0.0);
      std::array<double, 2> level{};
      for (std::size_t l = 0; l < 2; ++l) {
        const double t = kStep / static_cast<double>(1U << l);
        level[l] = (f(t) - (2.0 * f0) + f(-t)) / (t * t);
      }
      return richardson(level[0], level[1]);
    };
    const auto scaled = [](std::vector<double> v, const double t) {
      for (double &e : v) {
        e *= t;
      }
      return v;
    };
    std::array<double, 3> third_ladder{};
    std::array<double, 3> separate_ladder{};
    std::array<double, 3> total_ladder{};
    for (std::size_t e = 0; e < 3; ++e) {
      const Direction &dir = directions[e];
      third_ladder[e] =
          second([&](const double t) { return zg(ref_ints, scaled(dir.x, t), z_, Roots::Joint); });
      separate_ladder[e] = second(
          [&](const double t) { return zg(ref_ints, scaled(dir.x, t), z_, Roots::Separate); });
      total_ladder[e] = second([&](const double t) {
        if (t == 0.0) {
          return zg(ref_ints, zero, z_, Roots::Joint);
        }
        std::vector<fb::Atom> where = atoms_;
        for (const auto &[x, w] : dir.coordinates) {
          where = moved(std::move(where), x, w * t);
        }
        return zg(integrals(at(where)), scaled(dir.x, t), z_, Roots::Joint);
      });
    }
    const std::chrono::duration<double> ladder = std::chrono::steady_clock::now() - ladder_begin;

    // The mixed element from the sum's ladder and the two coordinates'.
    const auto mixed = [](const std::array<double, 3> &v) { return 0.5 * (v[2] - v[0] - v[1]); };
    std::array<double, 3> third_miss{};
    std::array<double, 3> total_miss{};
    std::array<double, 3> third_value{};
    std::array<double, 3> total_value{};
    double curvature_share = 0.0;
    for (std::size_t e = 0; e < 3; ++e) {
      third_value[e] = third_of(got, e);
      total_value[e] = total_of(got, e);
      third_miss[e] = std::abs(third_value[e] - third_ladder[e]);
      total_miss[e] = std::abs(total_value[e] - total_ladder[e]);
      EXPECT_LE(third_miss[e], kLadderTol) << "T(x_e, x_e, z) along " << directions[e].name << ": "
                                           << third_value[e] << " vs " << third_ladder[e];
      EXPECT_LE(total_miss[e], kLadderTol) << "the total along " << directions[e].name << ": "
                                           << total_value[e] << " vs " << total_ladder[e];
      curvature_share = std::max(curvature_share, std::abs(third_ladder[e] - separate_ladder[e]));
    }
    curvature_share =
        std::max(curvature_share, std::abs(mixed(third_ladder) - mixed(separate_ladder)));
    EXPECT_GT(curvature_share, kVisible) << "the root-set curvature's share";

    // Theta's share: the ladders at z less at z with Theta zeroed (F is
    // linear in z), along each direction, the third term's and the total's.
    double theta_share = 0.0;
    if (rotations_) {
      std::vector<double> theta_only(size_, 0.0);
      std::copy(z_.begin() + static_cast<std::ptrdiff_t>(size_ - n_theta_), z_.end(),
                theta_only.begin() + static_cast<std::ptrdiff_t>(size_ - n_theta_));
      for (std::size_t e = 0; e < 3; ++e) {
        const Direction &dir = directions[e];
        const double third = second([&](const double t) {
          return zg(ref_ints, scaled(dir.x, t), theta_only, Roots::Joint);
        });
        const double total = second([&](const double t) {
          if (t == 0.0) {
            return zg(ref_ints, zero, theta_only, Roots::Joint);
          }
          std::vector<fb::Atom> where = atoms_;
          for (const auto &[x, w] : dir.coordinates) {
            where = moved(std::move(where), x, w * t);
          }
          return zg(integrals(at(where)), scaled(dir.x, t), theta_only, Roots::Joint);
        });
        theta_share = std::max({theta_share, std::abs(third), std::abs(total)});
      }
      EXPECT_GT(theta_share, kVisible) << "Theta's share";
    }

    std::cout << std::format(
        "[ SA z QD  ] {} state {}: analytic {:.2f} s, ladders {:.1f} s\n"
        "[ SA z QD  ] T(x,x,z) A, B, A+B {:.6e} {:.6e} {:.6e}, misses {:.1e} {:.1e} {:.1e}; "
        "mixed {:.6e} miss {:.1e}\n"
        "[ SA z QD  ] total A, B, A+B {:.6e} {:.6e} {:.6e}, misses {:.1e} {:.1e} {:.1e}; mixed "
        "{:.6e} miss {:.1e}\n"
        "[ SA z QD  ] shares: root-set curvature {:.2e}, Theta {:.2e}; max |L_xy z| {:.2e}\n",
        name_, kState, seconds.count(), ladder.count(), third_value[0], third_value[1],
        third_value[2], third_miss[0], third_miss[1], third_miss[2], mixed(third_value),
        std::abs(mixed(third_value) - mixed(third_ladder)), total_value[0], total_value[1],
        total_value[2], total_miss[0], total_miss[1], total_miss[2], mixed(total_value),
        std::abs(mixed(total_value) - mixed(total_ladder)), curvature_share, theta_share,
        max_abs(got.lxy));
  }

  std::unique_ptr<fb::NodeResources> node_;
  std::optional<fb::Calculation> calc_;
  std::optional<fb::mcscf::Solution> solution_;
  std::optional<fb::BasisStamp> stamp_;
  std::optional<fb::BasisStamp> aux_stamp_;
  std::optional<fb::ci::DeviceStringSet> alpha_;
  std::optional<fb::ci::DeviceStringSet> beta_;
  std::optional<fb::ci::Rdm> rdm_;
  std::string name_;
  std::vector<double> weights_;
  std::vector<fb::Atom> atoms_;
  Geometry ref_;
  std::vector<double> c_host_;
  std::vector<double> roots_host_;
  std::vector<double> z_;
  std::vector<double> x_a_;
  std::vector<double> x_b_;
  std::vector<std::pair<std::size_t, std::size_t>> pairs_;
  double sa_energy_ = 0.0;
  bool rotations_ = false;
  std::size_t nroots_ = 0;
  std::size_t nmo_ = 0;
  std::size_t naux_ = 0;
  std::size_t nocc_ = 0;
  std::size_t dim_ = 0;
  std::size_t n_theta_ = 0;
  std::size_t size_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfSaCasscfZQd, UnequalWeightsMatchLadders) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  load("h2co_skew_ccpvdz_sa3_weighted");
  ASSERT_FALSE(HasFatalFailure());
  check();
}

TEST_F(HessianDfSaCasscfZQd, EqualWeightsMatchLadders) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  load("h2co_skew_ccpvdz_sa3");
  ASSERT_FALSE(HasFatalFailure());
  check();
}

} // namespace
