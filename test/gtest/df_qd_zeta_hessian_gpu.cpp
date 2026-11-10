// GPU cases for hessian::df_qd_zeta: the zeta term T = Sum_{r<s} zeta_sr
// <c_s|H|c_r> of a QD-NEVPT2 Lagrangian's Hessian (docs/nevpt2/nevpt2.tex
// Eq. qd-Lxx and the mixed and skeleton terms), on data/qd_nevpt2/'s
// h2co_skew_ccpvdz_sa3 (C1, three roots at equal weights) from this tree's
// own DF-SA-CASSCF through the registry, at the zeta of
// nevpt2::qd_sc_lagrangian's state K = 1.
//
// Each share against a Richardson ladder of T itself, evaluated on the host
// from this tree's integrals at each geometry along the symmetric connection
// C(R) = C (C^T S(R) C)^-1/2, the orbitals C(R) e^K and the root set's
// exponential e^A c_r (A = Sum_r (d_r c_r^T - c_r d_r^T) + Sum_{r<s}
// Theta_sr (c_s c_r^T - c_r c_s^T), every pair's rotation), the exponentials
// exact (their series), only the RDMs from ci::Rdm, and T by the
// polarization 1/2 (E(c + c S) - E(c - c S)), S = (n / 4) Z, at the weights
// 1 / n (the core energy multiplies |c_r +/- (c S)_r|^2, equal on both
// sides):
//
//     lxx      against d2/dt2 T(R_0, t u) along u, v and u + v, random
//              directions mixing orbital, out-of-span CI and rotation parts
//              (and the orbital and CI parts alone), never a ladder of a
//              Hessian product;
//     lxlambda against d/dR_x of dT/dt (R, t u) at t = 0, x in {A, B};
//     lxy      against d2/dt2 T(R_0 + t e, 0) along A, B and A + B.
//
// Visibility, each > 10x the bar on some element: every share; within lxx
// the orbital part (response::Hessian's), the orbital-CI part and the CI
// part, and the gradient term the orbital part carries because T is not
// stationary, Sum dT/dU . 1/2 {K_u, K_v}. Then the refusals.

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
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar, absolute, for each share against its ladder; a share is
// visible above ten times it.
constexpr double kLadderTol = 1e-7;
constexpr double kVisible = 10.0 * kLadderTol;
// The ladders' step, bohr along a coordinate and t along a direction, with
// half of it beside it (Richardson).
constexpr double kStep = 4e-3;
// The host energy at the reference against the roots' average.
constexpr double kEnergyTol = 1e-9;
// Terms of the exponentials' series: the arguments' norms stay below 0.2.
constexpr std::int32_t kSeriesTerms = 30;
// The reference SA-CASSCF's ||g|| and |Delta E|.
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;
// The QD state whose zeta is taken, and the two coordinates (O's z, the
// first H's y).
constexpr std::int32_t kState = 1;
constexpr std::size_t kCoordA = 5;
constexpr std::size_t kCoordB = 7;
// The random directions' seed and each block's norm.
constexpr std::uint32_t kSeed = 955;
constexpr double kBlockNorm = 1.0;

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

double dot(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
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

double richardson(const double coarse, const double fine) {
  return ((4 * fine) - coarse) / 3;
}

// d2/dt2 of f at 0 from f at -t, 0 and t, Richardson over kStep and half it.
template <typename F>
double second(const F &f) {
  const double f0 = f(0.0);
  std::array<double, 2> level{};
  for (std::size_t l = 0; l < 2; ++l) {
    const double t = kStep / static_cast<double>(1U << l);
    level[l] = (f(t) - (2.0 * f0) + f(-t)) / (t * t);
  }
  return richardson(level[0], level[1]);
}

// d/dt of f at 0, the same way.
template <typename F>
double first(const F &f) {
  std::array<double, 2> level{};
  for (std::size_t l = 0; l < 2; ++l) {
    const double t = kStep / static_cast<double>(1U << l);
    level[l] = (f(t) - f(-t)) / (2.0 * t);
  }
  return richardson(level[0], level[1]);
}

std::vector<double> scaled(std::vector<double> v, const double t) {
  for (double &e : v) {
    e *= t;
  }
  return v;
}

std::vector<double> added(std::vector<double> a, const std::vector<double> &b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] += b[i];
  }
  return a;
}

class HessianDfQdZeta : public ::testing::Test {
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

  // The case through the registry, then zeta and the random directions.
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
    equal_.assign(nroots_, 1.0 / static_cast<double>(nroots_));
    node_ = std::make_unique<fb::NodeResources>(
        fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces});
    const fb::test::ScratchDir dir("df_qd_zeta_hessian");
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
    // Each root's phase fixed, its largest element positive, so the random
    // directions (and zeta) mean the same from run to run.
    roots_host_ = cas->vectors;
    dim_ = roots_host_.size() / nroots_;
    for (std::size_t root = 0; root < nroots_; ++root) {
      const std::span<double> c(roots_host_.data() + (root * dim_), dim_);
      if (*std::ranges::max_element(c, {}, [](const double e) { return std::abs(e); }) < 0.0) {
        std::ranges::transform(c, c.begin(), std::negate{});
      }
    }
    upload(r, roots_host_, solution_->roots.vectors.data());
    ASSERT_EQ(wwr::wwrMemcpyAsync(solution_->c.data(), cas->c.data(),
                                  cas->c.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(r.sync().has_value());
    c_host_ = download(r, solution_->c.data(), nmo_ * nmo_);
    for (std::size_t root = 0; root < nroots_; ++root) {
      sa_energy_ += equal_[root] * cas->energies[root];
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
    n_theta_ = nroots_ * (nroots_ - 1) / 2;
    size_ = pairs_.size() + (dim_ * nroots_) + n_theta_;

    // zeta of the QD Lagrangian of state K, and S = (n / 4) Z.
    auto lagrangian = fb::nevpt2::qd_sc_lagrangian(r, molecule(), basis(), fit().aux, h(), fit().b,
                                                   *solution_, ncore_, ncas_, kState, 0);
    ASSERT_TRUE(lagrangian.has_value()) << lagrangian.error().message;
    zeta_ = lagrangian->pairs.zeta;
    ASSERT_EQ(zeta_.size(), n_theta_);
    max_gradient_ = max_abs(download(r, lagrangian->zeta.orbital.data(), pairs_.size()));
    s_.assign(nroots_ * nroots_, 0.0);
    std::size_t pair = 0;
    for (std::size_t a = 0; a < nroots_; ++a) {
      for (std::size_t b = a + 1; b < nroots_; ++b, ++pair) {
        s_[b + (nroots_ * a)] = s_[a + (nroots_ * b)] =
            zeta_[pair] * static_cast<double>(nroots_) / 4.0;
      }
    }

    // Two random directions, each block (orbital, CI off the roots, the
    // rotations) of norm kBlockNorm.
    std::mt19937 engine(kSeed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    const auto random = [&] {
      std::vector<double> v(size_);
      for (double &e : v) {
        e = uniform(engine);
      }
      v = off_roots(std::move(v));
      for (const auto &[begin, end] : blocks()) {
        double norm = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
          norm += v[i] * v[i];
        }
        for (std::size_t i = begin; i < end; ++i) {
          v[i] *= kBlockNorm / std::sqrt(norm);
        }
      }
      return v;
    };
    u_ = random();
    v_ = random();
    std::cout << std::format(
        "[ QD zeta   ] {}: E {}, ||g|| {:.1e}; zeta {}; max |zeta . g| {:.2e}\n", name,
        cas->energies, cas->gradient_norm, zeta_, max_gradient_);
    ref_ = at(atoms_);
    ref_ints_ = integrals(ref_);
    ASSERT_FALSE(HasFailure());
  }

  // The parameter blocks [begin, end): orbital, CI, rotations.
  [[nodiscard]] std::array<std::pair<std::size_t, std::size_t>, 3> blocks() const {
    const std::size_t ci = pairs_.size() + (dim_ * nroots_);
    return {std::pair{std::size_t{0}, pairs_.size()}, std::pair{pairs_.size(), ci},
            std::pair{ci, size_}};
  }

  // @p v with only block @p which kept.
  [[nodiscard]] std::vector<double> only(const std::vector<double> &v, std::size_t which) const {
    std::vector<double> out(size_, 0.0);
    const auto [begin, end] = blocks()[which];
    std::copy(v.begin() + static_cast<std::ptrdiff_t>(begin),
              v.begin() + static_cast<std::ptrdiff_t>(end),
              out.begin() + static_cast<std::ptrdiff_t>(begin));
    return out;
  }

  // @p v with each root's CI part taken off every root.
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

  // The root set's generator of a vector (dim x dim): Sum_r (d_r c_r^T - c_r
  // d_r^T) plus the rotations Sum_{r<s} Theta_sr (c_s c_r^T - c_r c_s^T).
  [[nodiscard]] std::vector<double> generator(const std::vector<double> &v) const {
    std::vector<double> a(dim_ * dim_, 0.0);
    const auto add = [&](const double *u, const double *w, const double scale) {
      for (std::size_t j = 0; j < dim_; ++j) {
        for (std::size_t i = 0; i < dim_; ++i) {
          a[i + (dim_ * j)] += scale * ((u[i] * w[j]) - (w[i] * u[j]));
        }
      }
    };
    for (std::size_t r = 0; r < nroots_; ++r) {
      add(v.data() + pairs_.size() + (r * dim_), roots_host_.data() + (r * dim_), 1.0);
    }
    const double *theta = v.data() + pairs_.size() + (dim_ * nroots_);
    std::size_t k = 0;
    for (std::size_t r = 0; r < nroots_; ++r) {
      for (std::size_t s = r + 1; s < nroots_; ++s, ++k) {
        add(roots_host_.data() + (s * dim_), roots_host_.data() + (r * dim_), theta[k]);
      }
    }
    return a;
  }

  // gamma (ncas^2) and Gamma (ncas^4), row-major, of the nroots vectors @p y
  // at the weights 1 / n.
  std::pair<std::vector<double>, std::vector<double>> rdms(const std::vector<double> &y) {
    fb::DeviceResources &r = res();
    const auto n = static_cast<std::size_t>(ncas_);
    fb::Buffers::device<double> v(y.size(), r.device());
    upload(r, y, v.data());
    fb::Buffers::device<double> out((n * n) + (n * n * n * n), r.device());
    EXPECT_TRUE(rdm_->rdm12(r, equal_, v.data(), out.data(), out.data() + (n * n)).has_value());
    const std::vector<double> all = download(r, out.data(), out.num_elements());
    return {std::vector<double>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n * n)),
            std::vector<double>(all.begin() + static_cast<std::ptrdiff_t>(n * n), all.end())};
  }

  // E_w(Y) = Sum_r (1 / n) <y_r|H|y_r> at the integrals @p ints and the
  // orbitals U (nmo x nmo), less V_nn, the core weighted by |y_r|^2; or with
  // @p du (and @p dy, null for none), its derivative along U + e dU and
  // Y + e dY: the orbital part by dE/dU_o at Y's densities, the CI part from
  // the difference of the densities of Y + dY and Y - dY. With @p minus (and
  // @p dminus beside @p dy), 1/2 (E_w(Y) - E_w(minus)) and its derivative,
  // every density differenced before it is contracted: T's ladders would
  // otherwise difference two energies of 145 hartree.
  double energy(const Integrals &ints, const std::vector<double> &u, const std::vector<double> &y,
                const std::vector<double> *minus = nullptr, const std::vector<double> *du = nullptr,
                const std::vector<double> *dy = nullptr,
                const std::vector<double> *dminus = nullptr) {
    const auto &[h_mo, l_mo] = ints;
    const std::size_t nmo = nmo_;
    const std::size_t nocc = nocc_;
    const std::size_t no2 = nocc * nocc;
    const auto core = static_cast<std::size_t>(ncore_);
    const auto n = static_cast<std::size_t>(ncas_);
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
    // Sum_r w_r |y_r|^2, the core's weight.
    const auto norm = [&](const std::vector<double> &vectors) {
      double n0 = 0.0;
      for (std::size_t r = 0; r < nroots_; ++r) {
        const double *yr = vectors.data() + (r * dim_);
        n0 += equal_[r] * std::inner_product(yr, yr + dim_, yr, 0.0);
      }
      return n0;
    };
    // D and Gamma over the occupied orbitals of the vectors at the weights,
    // the core weighted by @p n0.
    const auto full = [&](const std::vector<double> &vectors, const double n0) {
      const auto [g1, g2] = rdms(vectors);
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
    // full(a) - full(b) (b null: full(a)), times @p scale. Every pair this
    // differences has |a_r|^2 = |b_r|^2 (the polarized roots, and Y +/- dY
    // with dY = A Y, A antisymmetric), so both take a's core weight: the
    // core energy, 100 hartree, then cancels exactly instead of to its
    // rounding, which a ladder's 1 / t^2 would lift to 1e-8.
    const auto difference = [&](const std::vector<double> &a, const std::vector<double> *b,
                                const double scale) {
      const double n0 = norm(a);
      auto [da, ga] = full(a, n0);
      if (b != nullptr) {
        const auto [db, gb] = full(*b, n0);
        for (std::size_t i = 0; i < da.size(); ++i) {
          da[i] -= db[i];
        }
        for (std::size_t i = 0; i < ga.size(); ++i) {
          ga[i] -= gb[i];
        }
      }
      for (double &e : da) {
        e *= scale;
      }
      for (double &e : ga) {
        e *= scale;
      }
      return std::pair{std::move(da), std::move(ga)};
    };
    const double half = minus == nullptr ? 1.0 : 0.5;
    const auto [d, gg] = difference(y, minus, half);
    if (du == nullptr) {
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
      orbital += (*du)[i] * grad[i];
    }
    if (dy == nullptr) {
      return orbital;
    }
    // The densities of Y + dY less Y - dY, and with @p minus those of
    // minus + dminus less minus - dminus taken off them.
    const auto shifted = [](const std::vector<double> &a, const std::vector<double> &b,
                            const double sign) {
      std::vector<double> out = a;
      for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] += sign * b[i];
      }
      return out;
    };
    const std::vector<double> y_back = shifted(y, *dy, -1.0);
    auto [dc, gc] = difference(shifted(y, *dy, 1.0), &y_back, half);
    if (minus != nullptr) {
      const std::vector<double> minus_back = shifted(*minus, *dminus, -1.0);
      const auto [dm, gm] = difference(shifted(*minus, *dminus, 1.0), &minus_back, half);
      for (std::size_t i = 0; i < dc.size(); ++i) {
        dc[i] -= dm[i];
      }
      for (std::size_t i = 0; i < gc.size(); ++i) {
        gc[i] -= gm[i];
      }
    }
    double ci = 0.0;
    for (std::size_t i = 0; i < no2; ++i) {
      ci += ho[i] * dc[i];
    }
    for (std::size_t i = 0; i < no2 * no2; ++i) {
      ci += 0.5 * g[i] * gc[i];
    }
    return orbital + (0.5 * ci);
  }

  // The polarized pair Y (1 + S), Y (1 - S) of the roots @p y (dim x n).
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>>
  polarized(const std::vector<double> &y) const {
    const std::vector<double> ys = host_gemm(dim_, nroots_, nroots_, y, false, s_);
    std::vector<double> plus = y;
    std::vector<double> minus = y;
    for (std::size_t i = 0; i < y.size(); ++i) {
      plus[i] += ys[i];
      minus[i] -= ys[i];
    }
    return {std::move(plus), std::move(minus)};
  }

  // T at the integrals @p ints and the parameters @p lambda: the orbitals
  // e^K, the roots e^A c.
  double transition(const Integrals &ints, const std::vector<double> &lambda) {
    const std::vector<double> u = expm(kappa(lambda), nmo_);
    const std::vector<double> y =
        host_gemm(dim_, dim_, nroots_, expm(generator(lambda), dim_), false, roots_host_);
    const auto [plus, minus] = polarized(y);
    return energy(ints, u, plus, &minus);
  }

  // dT/dt (ints, t w) at t = 0, exactly: dU = K(w), dY = A(w) Y.
  double slope(const Integrals &ints, const std::vector<double> &w) {
    std::vector<double> one(nmo_ * nmo_, 0.0);
    for (std::size_t i = 0; i < nmo_; ++i) {
      one[i * (nmo_ + 1)] = 1.0;
    }
    const std::vector<double> du = kappa(w);
    const std::vector<double> a = generator(w);
    const auto [plus, minus] = polarized(roots_host_);
    const std::vector<double> dplus = host_gemm(dim_, dim_, nroots_, a, false, plus);
    const std::vector<double> dminus = host_gemm(dim_, dim_, nroots_, a, false, minus);
    return energy(ints, one, plus, &minus, &du, &dplus, &dminus);
  }

  // The gradient term the exponential carries along two orbital directions,
  // dT/dU . 1/2 (K_a K_b + K_b K_a): what a Hessian simplified at a
  // stationary point would drop.
  double gradient_term(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> one(nmo_ * nmo_, 0.0);
    for (std::size_t i = 0; i < nmo_; ++i) {
      one[i * (nmo_ + 1)] = 1.0;
    }
    const std::vector<double> ka = kappa(a);
    const std::vector<double> kb = kappa(b);
    const std::vector<double> du = scaled(added(host_gemm(nmo_, nmo_, nmo_, ka, false, kb),
                                                host_gemm(nmo_, nmo_, nmo_, kb, false, ka)),
                                          0.5);
    const auto [plus, minus] = polarized(roots_host_);
    return energy(ref_ints_, one, plus, &minus, &du);
  }

  // The atoms with coordinate x moved by @p step.
  [[nodiscard]] std::vector<fb::Atom> moved(std::vector<fb::Atom> atoms, std::size_t x,
                                            double step) const {
    fb::Vec<double, 3> &r = atoms[x / 3].position;
    (x % 3 == 0 ? r.x : x % 3 == 1 ? r.y : r.z) += step;
    return atoms;
  }

  // Every share against its ladder, and the pieces' visibility.
  void check() {
    fb::DeviceResources &r = res();
    const std::size_t n = n_x_;
    // The host energy is the roots' average at the reference.
    std::vector<double> one(nmo_ * nmo_, 0.0);
    for (std::size_t i = 0; i < nmo_; ++i) {
      one[i * (nmo_ + 1)] = 1.0;
    }
    EXPECT_NEAR(energy(ref_ints_, one, roots_host_),
                sa_energy_ - fb::nuclear_repulsion_energy(molecule()), kEnergyTol);

    // The analytic shares over the six block columns: u's orbital, CI and
    // rotation parts, then v's.
    std::vector<std::vector<double>> parts;
    for (const std::vector<double> *w : {&u_, &v_}) {
      for (std::size_t b = 0; b < 3; ++b) {
        parts.push_back(only(*w, b));
      }
    }
    const std::size_t k = parts.size();
    std::vector<double> lambda;
    for (const std::vector<double> &p : parts) {
      lambda.insert(lambda.end(), p.begin(), p.end());
    }
    fb::Buffers::device<double> lambda_dev(lambda.size(), r.device());
    upload(r, lambda, lambda_dev.data());
    const auto begin = std::chrono::steady_clock::now();
    auto got = fb::hessian::df_qd_zeta(r, molecule(), basis(), classes(), fit().aux, h(), fit().b,
                                       fit().l.data(), *solution_, ncore_, ncas_, zeta_,
                                       static_cast<std::int32_t>(k), lambda_dev.data());
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
    ASSERT_TRUE(got.has_value()) << got.error().message;
    ASSERT_EQ(got->lxx.size(), k * k);
    ASSERT_EQ(got->lxy.exy.size(), n * n);
    ASSERT_EQ(got->lxlambda.num_elements(), size_ * n);
    const std::vector<double> lxlambda = download(r, got->lxlambda.data(), size_ * n);
    double asymmetry = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      for (std::size_t i = 0; i < k; ++i) {
        asymmetry = std::max(asymmetry, std::abs(got->lxx[i + (k * j)] - got->lxx[j + (k * i)]));
      }
    }
    // T_xx between two directions given by their weights over the six
    // columns.
    const auto xx = [&](const std::array<double, 6> &a, const std::array<double, 6> &b) {
      double sum = 0.0;
      for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = 0; i < k; ++i) {
          sum += a[i] * b[j] * got->lxx[i + (k * j)];
        }
      }
      return sum;
    };

    // xx: three direction pairs, each against the ladder along a, b and
    // a + b: the whole directions, their orbital parts and their CI parts
    // (out-of-span and rotation).
    struct Pair {
      std::string name;
      std::array<double, 6> a;
      std::array<double, 6> b;
    };
    const std::array<Pair, 3> pairs{Pair{"whole", {1, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 1}},
                                    Pair{"orbital", {1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}},
                                    Pair{"CI", {0, 1, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 1}}};
    const auto vector_of = [&](const std::array<double, 6> &a) {
      std::vector<double> out(size_, 0.0);
      for (std::size_t i = 0; i < k; ++i) {
        out = added(std::move(out), scaled(parts[i], a[i]));
      }
      return out;
    };
    const auto ladder_begin = std::chrono::steady_clock::now();
    double xx_share = 0.0;
    for (const Pair &p : pairs) {
      std::array<double, 6> sum{};
      for (std::size_t i = 0; i < k; ++i) {
        sum[i] = p.a[i] + p.b[i];
      }
      const std::array<std::array<double, 6>, 3> along{p.a, p.b, sum};
      std::array<double, 3> analytic{};
      std::array<double, 3> ladder{};
      for (std::size_t e = 0; e < 3; ++e) {
        analytic[e] = xx(along[e], along[e]);
        const std::vector<double> w = vector_of(along[e]);
        ladder[e] = second([&](const double t) { return transition(ref_ints_, scaled(w, t)); });
        EXPECT_LE(std::abs(analytic[e] - ladder[e]), kLadderTol)
            << "T_xx along " << p.name << " direction " << e << ": " << analytic[e] << " vs "
            << ladder[e];
        if (p.name == "whole") {
          xx_share = std::max(xx_share, std::abs(analytic[e]));
        }
      }
      const double mixed = 0.5 * (analytic[2] - analytic[0] - analytic[1]);
      const double mixed_ladder = 0.5 * (ladder[2] - ladder[0] - ladder[1]);
      std::cout << std::format("[ QD zeta   ] xx {}: a, b, a+b {:.6e} {:.6e} {:.6e}, misses "
                               "{:.1e} {:.1e} {:.1e}; mixed {:.6e} miss {:.1e}\n",
                               p.name, analytic[0], analytic[1], analytic[2],
                               std::abs(analytic[0] - ladder[0]), std::abs(analytic[1] - ladder[1]),
                               std::abs(analytic[2] - ladder[2]), mixed,
                               std::abs(mixed - mixed_ladder));
    }

    // The pieces, over the elements (u, u), (v, v) and (u, v): the orbital
    // part, the orbital-CI part and the CI part; and the gradient term the
    // orbital part carries off stationarity.
    double orbital_piece = 0.0;
    double coupling_piece = 0.0;
    double ci_piece = 0.0;
    double gradient_piece = 0.0;
    const std::array<std::pair<std::size_t, std::size_t>, 3> elements{
        std::pair{std::size_t{0}, std::size_t{0}}, std::pair{std::size_t{3}, std::size_t{3}},
        std::pair{std::size_t{0}, std::size_t{3}}};
    for (const auto &[a, b] : elements) {
      const auto at = [&](std::size_t i, std::size_t j) {
        return got->lxx[(a + i) + (k * (b + j))];
      };
      orbital_piece = std::max(orbital_piece, std::abs(at(0, 0)));
      coupling_piece =
          std::max(coupling_piece, std::abs(at(0, 1) + at(0, 2) + at(1, 0) + at(2, 0)));
      ci_piece = std::max(ci_piece, std::abs(at(1, 1) + at(1, 2) + at(2, 1) + at(2, 2)));
      gradient_piece = std::max(gradient_piece, std::abs(gradient_term(parts[a], parts[b])));
    }
    EXPECT_GT(xx_share, kVisible) << "T_xx";
    EXPECT_GT(orbital_piece, kVisible) << "T_xx's orbital part";
    EXPECT_GT(coupling_piece, kVisible) << "T_xx's orbital-CI part";
    EXPECT_GT(ci_piece, kVisible) << "T_xx's CI part";
    EXPECT_GT(gradient_piece, kVisible) << "the orbital part's gradient term";

    // Ax: d/dR_x of dT/dt (R, t w) at fixed parameters, w = u and v, x = A
    // and B.
    double ax_share = 0.0;
    double ax_miss = 0.0;
    // Its orbital, CI and rotation rows' shares.
    std::array<double, 3> ax_blocks{};
    for (const std::size_t x : {kCoordA, kCoordB}) {
      for (const std::vector<double> *w : {&u_, &v_}) {
        const double ladder =
            first([&](const double t) { return slope(integrals(at(moved(atoms_, x, t))), *w); });
        double analytic = 0.0;
        for (std::size_t i = 0; i < size_; ++i) {
          analytic += lxlambda[i + (size_ * x)] * (*w)[i];
        }
        const double miss = std::abs(analytic - ladder);
        EXPECT_LE(miss, kLadderTol)
            << "L_Ax at coordinate " << x << ": " << analytic << " vs " << ladder;
        ax_share = std::max(ax_share, std::abs(analytic));
        ax_miss = std::max(ax_miss, miss);
        for (std::size_t block = 0; block < 3; ++block) {
          const auto [first_row, end_row] = blocks()[block];
          double part = 0.0;
          for (std::size_t i = first_row; i < end_row; ++i) {
            part += lxlambda[i + (size_ * x)] * (*w)[i];
          }
          ax_blocks[block] = std::max(ax_blocks[block], std::abs(part));
        }
      }
    }
    EXPECT_GT(ax_share, kVisible) << "L_Ax";
    for (std::size_t block = 0; block < 3; ++block) {
      EXPECT_GT(ax_blocks[block], kVisible) << "L_Ax's rows of block " << block;
    }

    // AB: d2/dt2 T(R_0 + t e, 0) along A, B and A + B.
    const std::vector<double> zero(size_, 0.0);
    struct Direction {
      std::string name;
      std::vector<std::pair<std::size_t, double>> coordinates;
    };
    const std::array<Direction, 3> directions{Direction{"A", {{kCoordA, 1.0}}},
                                              Direction{"B", {{kCoordB, 1.0}}},
                                              Direction{"A + B", {{kCoordA, 1.0}, {kCoordB, 1.0}}}};
    std::array<double, 3> ab_value{};
    std::array<double, 3> ab_ladder{};
    double ab_share = 0.0;
    for (std::size_t e = 0; e < 3; ++e) {
      const Direction &dir = directions[e];
      for (const auto &[x, wx] : dir.coordinates) {
        for (const auto &[y, wy] : dir.coordinates) {
          ab_value[e] += wx * wy * got->lxy.exy[x + (n * y)];
        }
      }
      ab_ladder[e] = second([&](const double t) {
        if (t == 0.0) {
          return transition(ref_ints_, zero);
        }
        std::vector<fb::Atom> where = atoms_;
        for (const auto &[x, w] : dir.coordinates) {
          where = moved(std::move(where), x, w * t);
        }
        return transition(integrals(at(where)), zero);
      });
      EXPECT_LE(std::abs(ab_value[e] - ab_ladder[e]), kLadderTol)
          << "L_AB along " << dir.name << ": " << ab_value[e] << " vs " << ab_ladder[e];
      ab_share = std::max(ab_share, std::abs(ab_value[e]));
    }
    EXPECT_GT(ab_share, kVisible) << "L_AB";
    const std::chrono::duration<double> ladders = std::chrono::steady_clock::now() - ladder_begin;
    const auto mixed = [](const std::array<double, 3> &w) { return 0.5 * (w[2] - w[0] - w[1]); };

    std::cout << std::format(
        "[ QD zeta   ] {}: analytic {:.2f} s, ladders {:.1f} s; xx asymmetry {:.1e}\n"
        "[ QD zeta   ] shares: xx {:.2e} (orbital {:.2e}, of it the gradient term {:.2e}; "
        "orbital-CI {:.2e}; CI {:.2e}), Ax {:.2e} (orbital {:.2e}, CI {:.2e}, rotations {:.2e}; "
        "miss {:.1e}), AB {:.2e}\n"
        "[ QD zeta   ] AB A, B, A+B {:.6e} {:.6e} {:.6e}, misses {:.1e} {:.1e} {:.1e}; mixed "
        "{:.6e} miss {:.1e}\n",
        name_, seconds.count(), ladders.count(), asymmetry, xx_share, orbital_piece, gradient_piece,
        coupling_piece, ci_piece, ax_share, ax_blocks[0], ax_blocks[1], ax_blocks[2], ax_miss,
        ab_share, ab_value[0], ab_value[1], ab_value[2], std::abs(ab_value[0] - ab_ladder[0]),
        std::abs(ab_value[1] - ab_ladder[1]), std::abs(ab_value[2] - ab_ladder[2]), mixed(ab_value),
        std::abs(mixed(ab_value) - mixed(ab_ladder)));

    // The refusals, before anything runs.
    const auto refused = [&](const fb::Result<fb::hessian::QdZetaTerms> &bad, const char *what) {
      ASSERT_FALSE(bad.has_value()) << what;
      EXPECT_EQ(bad.error().kind, fb::ErrorKind::InvalidConfig) << what;
      EXPECT_NE(bad.error().message.find("hessian::df_qd_zeta"), std::string::npos) << what;
    };
    const std::vector<double> short_zeta(zeta_.begin(), zeta_.end() - 1);
    refused(fb::hessian::df_qd_zeta(r, molecule(), basis(), classes(), fit().aux, h(), fit().b,
                                    fit().l.data(), *solution_, ncore_, ncas_, short_zeta),
            "zeta not one per pair");
    refused(fb::hessian::df_qd_zeta(r, molecule(), basis(), classes(), fit().aux, h(), fit().b,
                                    fit().l.data(), *solution_, ncore_, ncas_, zeta_, 2, nullptr),
            "no Lambda for k > 0");
    refused(fb::hessian::df_qd_zeta(r, molecule(), basis(), classes(), fit().aux, h(), fit().b,
                                    fit().l.data(), *solution_, ncore_,
                                    ncas_ + static_cast<std::int32_t>(nmo_), zeta_),
            "ncore + ncas past nmo");
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
  std::vector<double> equal_;
  std::vector<fb::Atom> atoms_;
  Geometry ref_;
  Integrals ref_ints_;
  std::vector<double> c_host_;
  std::vector<double> roots_host_;
  std::vector<double> zeta_;
  std::vector<double> s_;
  std::vector<double> u_;
  std::vector<double> v_;
  std::vector<std::pair<std::size_t, std::size_t>> pairs_;
  double sa_energy_ = 0.0;
  double max_gradient_ = 0.0;
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

TEST_F(HessianDfQdZeta, EqualWeightsMatchLadders) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  load("h2co_skew_ccpvdz_sa3");
  ASSERT_FALSE(HasFatalFailure());
  check();
}

} // namespace
