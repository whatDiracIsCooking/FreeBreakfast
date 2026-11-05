// GPU cases for hessian::df_adjoint_xy, Sum h_bar h^xy + Sum Lambda B^xy at
// a general symmetric MO adjoint along the orthonormal path and the fitted
// index's symmetric gauge. On a skew water in cc-pVDZ over cc-pVDZ-JKFIT, at
// random orthonormal orbitals and a random symmetric (h_bar, Lambda): every
// coordinate pair against a fourth-order ladder of mcscf::df_gradient's
// extra-adjoint pass (df_transition_gradient at zero RDMs, so the pass alone)
// at the adjoint held fixed, the orbitals and the factor carried to each
// displaced geometry along the same path; the memory refusal, and a run two
// coordinates a chunk; then the input refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar: every pair within 1e-7 of the ladder's largest element.
constexpr double kLadderTol = 1e-7;
// The ladder's displacement (bohr), and twice it beside it for the report:
// a fourth-order stencil's miss falls 16x a halving.
constexpr double kStep = 2.5e-4;

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *d) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(d, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
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

std::vector<double> transpose(const std::vector<double> &a, std::size_t n) {
  std::vector<double> t(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      t[j + (n * i)] = a[i + (n * j)];
    }
  }
  return t;
}

// A^-1/2 of a symmetric A near the identity (||1 - A|| < 1), by the coupled
// Newton-Schulz iteration: Y -> A^1/2, Z -> A^-1/2.
std::vector<double> inverse_sqrt(const std::vector<double> &a, std::size_t n) {
  std::vector<double> y = a;
  std::vector<double> z(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z[i * (n + 1)] = 1.0;
  }
  for (std::int32_t it = 0; it < 30; ++it) {
    std::vector<double> t = host_gemm(n, n, n, z, false, y);
    double off = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      off = std::max(off, std::abs(t[i] - (i % (n + 1) == 0 ? 1.0 : 0.0)));
      t[i] = -0.5 * t[i];
    }
    if (off < 1e-15) {
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

// L^-1 B for the lower triangle of the n x n @p l and the n x cols @p b.
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

// The eigenvalues and eigenvectors (columns) of the symmetric n x n @p a by
// cyclic Jacobi.
std::pair<std::vector<double>, std::vector<double>> eigen(std::vector<double> a, std::size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    v[i * (n + 1)] = 1.0;
  }
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        off = std::max(off, std::abs(a[p + (n * q)]));
      }
    }
    if (off < 1e-15) {
      break;
    }
    for (std::size_t q = 1; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        const double apq = a[p + (n * q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double theta = (a[q * (n + 1)] - a[p * (n + 1)]) / (2.0 * apq);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double cs = 1.0 / std::sqrt((t * t) + 1.0);
        const double sn = t * cs;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k + (n * p)];
          const double akq = a[k + (n * q)];
          a[k + (n * p)] = (cs * akp) - (sn * akq);
          a[k + (n * q)] = (sn * akp) + (cs * akq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[p + (n * k)];
          const double aqk = a[q + (n * k)];
          a[p + (n * k)] = (cs * apk) - (sn * aqk);
          a[q + (n * k)] = (sn * apk) + (cs * aqk);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double vkp = v[k + (n * p)];
          const double vkq = v[k + (n * q)];
          v[k + (n * p)] = (cs * vkp) - (sn * vkq);
          v[k + (n * q)] = (sn * vkp) + (cs * vkq);
        }
      }
    }
  }
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[i * (n + 1)];
  }
  return {values, v};
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// The skew water, bohr: no symmetry for a zero to hide behind.
std::vector<fb::Atom> skew_water() {
  return {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.12}, static_cast<fb::Element>(8)},
          fb::Atom{fb::Vec<double, 3>{1.45, 0.08, -0.95}, static_cast<fb::Element>(1)},
          fb::Atom{fb::Vec<double, 3>{-1.38, 0.21, -1.02}, static_cast<fb::Element>(1)}};
}

// The integrals at one geometry as the DF steps would build them: the bases
// placed, h, the fit B and its L on the device, and S, h, the metric and L
// on the host (L's strict upper triangle zeroed).
class Geometry {
public:
  Geometry(fb::DeviceResources &res, const std::vector<fb::Atom> &atoms) : res_(res) {
    molecule_.emplace(res, atoms);
    const auto stamp = fb::resolve_basis("cc-pvdz");
    const auto aux_stamp = fb::resolve_aux_basis("cc-pvdz-jkfit");
    EXPECT_TRUE(stamp && aux_stamp);
    if (!stamp || !aux_stamp) {
      return;
    }
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    EXPECT_TRUE(basis && aux);
    if (!basis || !aux) {
      return;
    }
    basis_.emplace(std::move(*basis));
    aux_.emplace(std::move(*aux));
    nao_ = basis_->nao_sphe();
    naux_ = aux_->nao_sphe();
    auto classes = fb::primitive_pair_classes(res, *basis_);
    auto basis_u = fb::upload_spherical_transform_matrix(res, *basis_);
    auto aux_u = fb::upload_spherical_transform_matrix(res, *aux_);
    EXPECT_TRUE(classes && basis_u && aux_u);
    if (!classes || !basis_u || !aux_u) {
      return;
    }
    classes_ = std::move(*classes);
    const std::size_t nn = nao_ * nao_;
    fb::MatrixAOPair s_pair(res, *basis_, *basis_u);
    EXPECT_TRUE(fb::overlap(res, classes_, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(res));
    s_ = download(res, s_pair.spherical().data(), nn);
    h_.emplace(res, *basis_, std::move(*basis_u));
    EXPECT_TRUE(fb::core_hamiltonian(res, classes_, *molecule_, h_->cartesian()));
    EXPECT_TRUE(h_->cartesian_to_spherical(res));
    h_host_ = download(res, h_->spherical().data(), nn);
    metric_.emplace(res, *aux_, std::move(*aux_u));
    EXPECT_TRUE(fb::metric(res, *aux_, *metric_));
    m_ = download(res, metric_->spherical().data(), naux_ * naux_);
    b_.emplace(naux_ * nn, res.device());
    EXPECT_TRUE(fb::three_center(res, *basis_, classes_, *aux_, b_->data()));
    EXPECT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nao_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));
    l_ = download(res, metric_->spherical().data(), naux_ * naux_);
    for (std::size_t j = 0; j < naux_; ++j) {
      for (std::size_t i = 0; i < j; ++i) {
        l_[i + (naux_ * j)] = 0.0;
      }
    }
  }

  [[nodiscard]] bool ok() const { return b_.has_value(); }

  // C^T S C's inverse square root applied to @p c0: the orbitals carried
  // here along the connection.
  [[nodiscard]] std::vector<double> carried(const std::vector<double> &c0) const {
    const std::size_t n = nao_;
    return host_gemm(
        n, n, n, c0, false,
        inverse_sqrt(host_gemm(n, n, n, c0, true, host_gemm(n, n, n, s_, false, c0)), n));
  }

  fb::DeviceResources &res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> h_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::vector<double> s_;
  std::vector<double> h_host_;
  std::vector<double> m_;
  std::vector<double> l_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
};

// The reference: its integrals, random orthonormal orbitals C0 = S^-1/2 U
// (U the eigenvectors of a random symmetric matrix), and a random
// symmetric adjoint, elements uniform in [-1, 1], on the host and device.
class HessianDfAdjointXy : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    res_.emplace(0);
    atoms_ = skew_water();
    ref_.emplace(*res_, atoms_);
    ASSERT_TRUE(ref_->ok());
    nmo_ = ref_->nao_;
    naux_ = ref_->naux_;
    nn_ = nmo_ * nmo_;
    n_x_ = 3 * atoms_.size();
    std::mt19937_64 rng(809);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<double> random(nn_);
    for (std::size_t j = 0; j < nmo_; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        random[i + (nmo_ * j)] = random[j + (nmo_ * i)] = uniform(rng);
      }
    }
    const auto [s_values, s_vectors] = eigen(ref_->s_, nmo_);
    std::vector<double> scaled = s_vectors;
    for (std::size_t j = 0; j < nmo_; ++j) {
      for (std::size_t i = 0; i < nmo_; ++i) {
        scaled[i + (nmo_ * j)] /= std::sqrt(s_values[j]);
      }
    }
    const std::vector<double> s_inv_half =
        host_gemm(nmo_, nmo_, nmo_, scaled, false, transpose(s_vectors, nmo_));
    c0_ = host_gemm(nmo_, nmo_, nmo_, s_inv_half, false, eigen(random, nmo_).second);
    h_bar_.assign(nn_, 0.0);
    lambda_.assign(naux_ * nn_, 0.0);
    for (std::size_t j = 0; j < nmo_; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        h_bar_[i + (nmo_ * j)] = h_bar_[j + (nmo_ * i)] = uniform(rng);
        for (std::size_t q = 0; q < naux_; ++q) {
          lambda_[q + (naux_ * (i + (nmo_ * j)))] = lambda_[q + (naux_ * (j + (nmo_ * i)))] =
              uniform(rng);
        }
      }
    }
    c_.emplace(*res_, *ref_->basis_);
    upload(*res_, c0_, c_->data());
    h_bar_dev_.emplace(nn_, res_->device());
    upload(*res_, h_bar_, h_bar_dev_->data());
    lambda_dev_.emplace(naux_ * nn_, res_->device());
    upload(*res_, lambda_, lambda_dev_->data());
  }

  fb::Result<std::vector<double>> run(const fb::BasisSet &aux, const fb::SphericalMatrixAO &c,
                                      const double *h_bar, const double *lambda,
                                      const std::size_t max_bytes = 0) {
    return fb::hessian::df_adjoint_xy(
        *res_, *ref_->molecule_, *ref_->basis_, ref_->classes_, aux, ref_->h_->spherical(),
        *ref_->b_, ref_->metric_->spherical().data(), c, h_bar, lambda, max_bytes);
  }

  // The extra-adjoint pass of mcscf::df_gradient at the reference's atoms
  // with coordinate @p y moved by @p step: the orbitals carried there, B's
  // fitted index in the reference's symmetric gauge, the adjoint held. The
  // Cholesky factor density_fit takes there is B_sym = O B_chol, O =
  // (L0^-1 M L0^-T)^-1/2 L0^-1 L, so Lambda goes over as O^T Lambda (the
  // symmetric gauge's derivative is the same from either frame), and X is
  // 2 h_MO h_bar + 2 Sum B_Q Lambda_Q there. Zero RDMs over one active
  // orbital and no core leave the transition gradient the pass alone.
  std::vector<double> pass(const std::size_t y, const double step) {
    std::vector<fb::Atom> atoms = atoms_;
    fb::Vec<double, 3> &moved = atoms[y / 3].position;
    (y % 3 == 0 ? moved.x : y % 3 == 1 ? moved.y : moved.z) += step;
    const Geometry at(*res_, atoms);
    EXPECT_TRUE(at.ok());
    if (!at.ok()) {
      return {};
    }
    const std::vector<double> c = at.carried(c0_);
    const std::vector<double> &l0 = ref_->l_;
    const std::vector<double> half = lower_solve(l0, naux_, at.m_, naux_);
    const std::vector<double> gm = lower_solve(l0, naux_, transpose(half, naux_), naux_);
    const std::vector<double> o = host_gemm(naux_, naux_, naux_, inverse_sqrt(gm, naux_), false,
                                            lower_solve(l0, naux_, at.l_, naux_));
    const std::vector<double> lambda = host_gemm(naux_, naux_, nn_, o, true, lambda_);

    // X: h_MO h_bar, and Sum_Q B_Q Lambda'_Q over the Cholesky B in the MO
    // basis.
    const std::vector<double> b_ao = download(*res_, at.b_->data(), naux_ * nn_);
    std::vector<double> x = host_gemm(
        nmo_, nmo_, nmo_,
        host_gemm(nmo_, nmo_, nmo_, c, true, host_gemm(nmo_, nmo_, nmo_, at.h_host_, false, c)),
        false, h_bar_);
    std::vector<double> slab(nn_);
    std::vector<double> lam(nn_);
    for (std::size_t q = 0; q < naux_; ++q) {
      for (std::size_t i = 0; i < nn_; ++i) {
        slab[i] = b_ao[q + (naux_ * i)];
        lam[i] = lambda[q + (naux_ * i)];
      }
      const std::vector<double> b_mo =
          host_gemm(nmo_, nmo_, nmo_, c, true, host_gemm(nmo_, nmo_, nmo_, slab, false, c));
      const std::vector<double> product = host_gemm(nmo_, nmo_, nmo_, b_mo, false, lam);
      for (std::size_t i = 0; i < nn_; ++i) {
        x[i] += product[i];
      }
    }
    for (double &v : x) {
      v *= 2.0;
    }

    fb::Buffers::device<double> adjoint((2 * nn_) + (naux_ * nn_), res_->device());
    upload(*res_, x, adjoint.data());
    upload(*res_, lambda, adjoint.data() + nn_);
    upload(*res_, h_bar_, adjoint.data() + nn_ + (naux_ * nn_));
    const fb::mcscf::MoAdjoint extra{.h_bar = adjoint.data() + nn_ + (naux_ * nn_),
                                     .lambda = adjoint.data() + nn_,
                                     .x = adjoint.data()};
    fb::mcscf::Solution solution{
        .c = fb::SphericalMatrixAO(*res_, *at.basis_),
        .roots = {.energies = {0.0},
                  .spin_squares = {0.0},
                  .vectors = fb::Buffers::device<double>(1, res_->device()),
                  .davidson = {},
                  .fp32_iterations = 0},
        .energy = 0.0,
        .gradient_norm = 0.0,
        .gamma = {0.0},
        .gamma2 = {0.0},
        .root_gammas = {0.0},
        .iterations = {},
        .weights = {1.0}};
    upload(*res_, c, solution.c.data());
    const std::vector<double> zero{0.0};
    auto terms = fb::mcscf::df_transition_gradient(
        *res_, *at.molecule_, *at.basis_, at.classes_, *at.aux_, at.h_->spherical(), *at.b_,
        at.metric_->spherical().data(), solution, 0, 1, zero, zero, nullptr, &extra);
    EXPECT_TRUE(terms.has_value()) << terms.error().message;
    return terms.has_value() ? terms->total : std::vector<double>{};
  }

  // The fourth-order central difference of the pass along every coordinate
  // at step @p h: column y of J, J[x + n_x y] = d g_x / d y.
  std::vector<double> ladder(const double h) {
    std::vector<double> j(n_x_ * n_x_);
    for (std::size_t y = 0; y < n_x_; ++y) {
      const std::vector<double> p1 = pass(y, h);
      const std::vector<double> m1 = pass(y, -h);
      const std::vector<double> p2 = pass(y, 2.0 * h);
      const std::vector<double> m2 = pass(y, -2.0 * h);
      if (p1.size() != n_x_ || m1.size() != n_x_ || p2.size() != n_x_ || m2.size() != n_x_) {
        ADD_FAILURE() << "a displaced pass failed along " << y;
        return {};
      }
      for (std::size_t x = 0; x < n_x_; ++x) {
        j[x + (n_x_ * y)] = ((8.0 * (p1[x] - m1[x])) - (p2[x] - m2[x])) / (12.0 * h);
      }
    }
    return j;
  }

  std::optional<fb::DeviceResources> res_;
  std::vector<fb::Atom> atoms_;
  std::optional<Geometry> ref_;
  std::optional<fb::SphericalMatrixAO> c_;
  std::optional<fb::Buffers::device<double>> h_bar_dev_;
  std::optional<fb::Buffers::device<double>> lambda_dev_;
  std::vector<double> c0_;
  std::vector<double> h_bar_;
  std::vector<double> lambda_;
  std::size_t nmo_ = 0;
  std::size_t naux_ = 0;
  std::size_t nn_ = 0;
  std::size_t n_x_ = 0;
};

// Every coordinate pair against the ladder's symmetric part. The pass at a
// displaced geometry differentiates along the connection and the gauge
// re-centred there, whose tangent differs from the path's by an
// antisymmetric 1/8 [S^x, S^y] (and 1/8 [m^x, m^y] on the fitted index) at
// first order in the step: J = H + K with K antisymmetric in (x, y), seen
// through X's and G's antisymmetric parts (README.md). So H is checked
// against (J + J^T) / 2, which also checks its own symmetry. The analytic
// run is the budget's smallest that takes two coordinates a chunk, after
// the refusal below one.
TEST_F(HessianDfAdjointXy, MatchesTheLadder) {
  ASSERT_FALSE(HasFatalFailure());
  const auto tiny = run(*ref_->aux_, *c_, h_bar_dev_->data(), lambda_dev_->data(), 1);
  ASSERT_FALSE(tiny.has_value());
  ASSERT_EQ(tiny.error().kind, fb::ErrorKind::Unsupported) << tiny.error().message;
  const std::string &message = tiny.error().message;
  const std::size_t needs = message.find(" need ");
  ASSERT_NE(needs, std::string::npos) << message;
  const std::size_t need = std::stoull(message.substr(needs + 6));
  const auto start = std::chrono::steady_clock::now();
  const auto got = run(*ref_->aux_, *c_, h_bar_dev_->data(), lambda_dev_->data(),
                       need + (sizeof(double) * naux_ * nn_));
  ASSERT_TRUE(got.has_value()) << got.error().message;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_EQ(got->size(), n_x_ * n_x_);
  const auto whole = run(*ref_->aux_, *c_, h_bar_dev_->data(), lambda_dev_->data());
  ASSERT_TRUE(whole.has_value()) << whole.error().message;

  const auto ladder_start = std::chrono::steady_clock::now();
  const std::vector<double> j = ladder(kStep);
  const std::vector<double> j_coarse = ladder(2.0 * kStep);
  ASSERT_FALSE(HasFailure());
  const double ladder_elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - ladder_start).count();
  const auto symmetric = [this](const std::vector<double> &a) {
    std::vector<double> s(a.size());
    for (std::size_t y = 0; y < n_x_; ++y) {
      for (std::size_t x = 0; x < n_x_; ++x) {
        s[x + (n_x_ * y)] = 0.5 * (a[x + (n_x_ * y)] + a[y + (n_x_ * x)]);
      }
    }
    return s;
  };
  const std::vector<double> ref = symmetric(j);
  const std::vector<double> ref_coarse = symmetric(j_coarse);
  const double scale = max_abs(ref);
  double worst = 0.0;
  double worst_coarse = 0.0;
  double curvature = 0.0;
  double chunk_drift = 0.0;
  std::size_t worst_at = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double miss = std::abs((*got)[i] - ref[i]);
    if (miss > worst) {
      worst = miss;
      worst_at = i;
    }
    worst_coarse = std::max(worst_coarse, std::abs((*got)[i] - ref_coarse[i]));
    curvature = std::max(curvature, std::abs(j[i] - ref[i]));
    chunk_drift = std::max(chunk_drift, std::abs((*got)[i] - (*whole)[i]));
  }
  EXPECT_LE(worst, kLadderTol * scale)
      << "at (" << worst_at % n_x_ << ", " << worst_at / n_x_ << ")";
  std::cout << std::format(
      "[ adjoint  ] skew water cc-pVDZ/cc-pVDZ-JKFIT, {} orbitals, {} fitting functions: max "
      "|H| {:.3e}, |H - ladder| {:.2e} ({:.2e} relative) at step {:g}, {:.2e} at {:g}; the "
      "ladder's antisymmetric part {:.2e}; chunked vs whole {:.1e}; {:.2f} s ({} bytes, two "
      "coordinates a chunk), ladder {:.1f} s\n",
      nmo_, naux_, scale, worst, worst / scale, kStep, worst_coarse, 2.0 * kStep, curvature,
      chunk_drift, elapsed, need + (sizeof(double) * naux_ * nn_), ladder_elapsed);
}

TEST_F(HessianDfAdjointXy, Refusals) {
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<std::vector<double>> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_adjoint_xy"), std::string::npos) << what;
  };
  expect_refused(run(*ref_->aux_, *c_, nullptr, lambda_dev_->data()), "no h_bar");
  expect_refused(run(*ref_->aux_, *c_, h_bar_dev_->data(), nullptr), "no Lambda");
  // The fitting set on two of the three atoms.
  const fb::Molecule two(*res_, {atoms_[0], atoms_[1]});
  const auto aux_stamp = fb::resolve_aux_basis("cc-pvdz-jkfit");
  ASSERT_TRUE(aux_stamp);
  const auto aux_two = fb::place(aux_stamp->primitive_normalized(), two);
  ASSERT_TRUE(aux_two);
  expect_refused(run(*aux_two, *c_, h_bar_dev_->data(), lambda_dev_->data()), "aux on other atoms");
  // C over another basis: the fitting set's functions as orbitals.
  const fb::SphericalMatrixAO wrong(*res_, *ref_->aux_);
  expect_refused(run(*ref_->aux_, wrong, h_bar_dev_->data(), lambda_dev_->data()),
                 "C not nao x nao");
}

} // namespace
