/// @file
/// @brief relaxed_hessian: H'(x) and its adjoint on the host, the CI response
/// by conjugate gradients over the sigma, and H~ x = H x + R x.

module free_breakfast.mcscf.relaxed_hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, geam, axpy, dot, nrm2, scal, gpu_status
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The CI response's conjugate gradients stop at a residual this fraction of
// the right-hand side's, or after this many sigmas per root. Two roots closer
// than kResponseGapFloor (hartree) leave their mixing out.
constexpr double kResponseTolerance = 1e-8;
constexpr int32_t kResponseMaxIterations = 200;
constexpr double kResponseGapFloor = 1e-8;

void response_integrals(const ResponseIntegrals &ints, std::span<const double> kappa,
                        std::span<double> h1, std::span<double> eri1) {
  const size_t mo = ints.nmo;
  const size_t core = ints.ncore;
  const size_t n = ints.ncas;
  const auto at = [n](size_t t, size_t u, size_t v, size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  std::vector<double> x(n * n * n * n, 0.0);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          double sum = 0.0;
          for (size_t p = 0; p < mo; ++p) {
            sum += kappa[p + (mo * (core + t))] * ints.pq_uv(p, core + u, v, w);
          }
          x[at(t, u, v, w)] = sum;
        }
      }
    }
  }
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri1[at(t, u, v, w)] =
              x[at(t, u, v, w)] + x[at(u, t, v, w)] + x[at(v, w, t, u)] + x[at(w, v, t, u)];
        }
      }
      double sum = 0.0;
      for (size_t p = 0; p < mo; ++p) {
        sum += (kappa[p + (mo * (core + t))] * ints.fi(p, core + u)) +
               (kappa[p + (mo * (core + u))] * ints.fi(core + t, p));
        for (size_t i = 0; i < core; ++i) {
          sum += kappa[p + (mo * i)] *
                 ((4.0 * ints.pq_uv(p, i, t, u)) - ints.pu_qv(p, i, t, u) - ints.pu_qv(p, i, u, t));
        }
      }
      h1[(t * n) + u] = sum;
    }
  }
}

void response_gradient(const ResponseIntegrals &ints, std::span<const double> gamma,
                       std::span<const double> gamma2, std::span<double> a) {
  const size_t mo = ints.nmo;
  const size_t core = ints.ncore;
  const size_t n = ints.ncas;
  const auto at = [n](size_t t, size_t u, size_t v, size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  // d/dX of the two-electron part: X enters (tu|vw)' four times.
  std::vector<double> y(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          y[at(t, u, v, w)] = gamma2[at(t, u, v, w)] + gamma2[at(u, t, v, w)] +
                              gamma2[at(v, w, t, u)] + gamma2[at(v, w, u, t)];
        }
      }
    }
  }
  std::vector<double> g(mo * mo, 0.0);
  for (size_t p = 0; p < mo; ++p) {
    for (size_t t = 0; t < n; ++t) {
      double sum = 0.0;
      for (size_t u = 0; u < n; ++u) {
        sum += (gamma[(t * n) + u] * ints.fi(p, core + u)) +
               (gamma[(u * n) + t] * ints.fi(core + u, p));
        for (size_t v = 0; v < n; ++v) {
          for (size_t w = 0; w < n; ++w) {
            sum += y[at(t, u, v, w)] * ints.pq_uv(p, core + u, v, w);
          }
        }
      }
      g[p + (mo * (core + t))] += sum;
    }
    for (size_t i = 0; i < core; ++i) {
      double sum = 0.0;
      for (size_t t = 0; t < n; ++t) {
        for (size_t u = 0; u < n; ++u) {
          sum += gamma[(t * n) + u] *
                 ((4.0 * ints.pq_uv(p, i, t, u)) - ints.pu_qv(p, i, t, u) - ints.pu_qv(p, i, u, t));
        }
      }
      g[p + (mo * i)] += sum;
    }
  }
  for (size_t q = 0; q < mo; ++q) {
    for (size_t p = 0; p < mo; ++p) {
      a[p + (mo * q)] = g[p + (mo * q)] - g[q + (mo * p)];
    }
  }
}

Result<RelaxedHessian>
RelaxedHessian::create(DeviceResources &res, response::Hessian &hessian, ci::Sigma &sigma,
                       ci::Rdm &rdm, const ci::DeviceStringSet &alpha,
                       const ci::DeviceStringSet &beta, const ci::Roots &roots,
                       std::span<const double> weights, ResponseIntegrals ints) {
  const size_t mo = ints.nmo;
  const size_t core = ints.ncore;
  const size_t n = ints.ncas;
  if (ints.fock_inactive.size() != mo * mo || ints.ppaa.size() != mo * mo * n * n ||
      ints.papa.size() != mo * mo * n * n || core + n > mo) {
    return err_config("mcscf::RelaxedHessian: needs F^I nmo^2 and ppaa, papa nmo^2 ncas^2 long, "
                      "ncore + ncas <= nmo (nmo {}, ncore {}, ncas {}; got {}, {}, {})",
                      mo, core, n, ints.fock_inactive.size(), ints.ppaa.size(), ints.papa.size());
  }
  const size_t pairs = (core * (mo - core)) + (n * (mo - core - n));
  if (static_cast<size_t>(hessian.size()) != pairs) {
    return err_config("mcscf::RelaxedHessian: H is over {} pairs, the integrals' orbitals {}",
                      hessian.size(), pairs);
  }
  const size_t roots_count = roots.energies.size();
  const auto dim = static_cast<size_t>(sigma.size());
  if (roots_count == 0 || weights.size() != roots_count ||
      roots.vectors.num_elements() < dim * roots_count) {
    return err_config("mcscf::RelaxedHessian: needs one weight per root and the roots as that many "
                      "vectors of {} (roots {}, weights {}, {} doubles)",
                      dim, roots_count, weights.size(), roots.vectors.num_elements());
  }
  ci::Sigma response_sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  return RelaxedHessian{hessian,
                        sigma,
                        rdm,
                        roots,
                        std::vector<double>(weights.begin(), weights.end()),
                        std::move(ints),
                        std::move(response_sigma),
                        dim,
                        res};
}

RelaxedHessian::RelaxedHessian(response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm,
                               const ci::Roots &roots, std::vector<double> weights,
                               ResponseIntegrals ints, ci::Sigma response_sigma, const size_t dim,
                               DeviceResources &res)
    : hessian_(&hessian), sigma_(&sigma), rdm_(&rdm), roots_(&roots), weights_(std::move(weights)),
      ints_(std::move(ints)), n_pairs_(hessian.size()), dim_(dim),
      response_sigma_(std::move(response_sigma)), ci_b_(dim * weights_.size(), res.device()),
      ci_delta_(dim * weights_.size(), res.device()),
      ci_pm_(2 * dim * weights_.size(), res.device()), cg_vectors_(3 * dim, res.device()),
      ci_overlaps_(weights_.size(), res.device()),
      response_matrix_(ints_.nmo * ints_.nmo, res.device()),
      response_packed_(std::max<size_t>(static_cast<size_t>(n_pairs_), 1), res.device()),
      response_gamma_(2 * ints_.ncas * ints_.ncas, res.device()),
      response_gamma2_(2 * ints_.ncas * ints_.ncas * ints_.ncas * ints_.ncas, res.device()) {}

Status RelaxedHessian::respond(DeviceResources &res, const double *const x,
                               std::span<double> gamma_t, std::span<double> gamma2_t) {
  if (gamma_t.size() != ints_.ncas * ints_.ncas ||
      gamma2_t.size() != gamma_t.size() * gamma_t.size()) {
    return err_config("mcscf::RelaxedHessian::respond: gamma_t and gamma2_t must be ncas^2 and "
                      "ncas^4 long (ncas {}; got {}, {})",
                      ints_.ncas, gamma_t.size(), gamma2_t.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t mo = ints_.nmo;
  const size_t n = ints_.ncas;
  const auto nmo = static_cast<int32_t>(mo);
  const auto ncore = static_cast<int32_t>(ints_.ncore);
  const auto ncas = static_cast<int32_t>(n);
  const size_t dim = dim_;
  const auto dim32 = static_cast<int32_t>(dim);
  const size_t roots_count = weights_.size();
  const auto nroots = static_cast<int32_t>(roots_count);
  const size_t ci_block = dim * roots_count;
  const double *const ci_c = roots_->vectors.data();
  const auto ci_root = [dim](auto *base, const size_t r) { return base + (r * dim); };
  // v <- v - C (C^T v): off every root.
  const auto ci_project = [&](double *v) -> Status {
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, dim32, nroots,
                                                  &kOne<double>, ci_c, dim32, v, 1, &kZero<double>,
                                                  ci_overlaps_.data(), 1)));
    return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, dim32, nroots,
                                                 &kNegOne<double>, ci_c, dim32, ci_overlaps_.data(),
                                                 1, &kOne<double>, v, 1));
  };

  // H'(x) from kappa(x), and b_r = H' c_r for every root.
  FBF_TRY(response::unpack_orbital_pairs(res, nmo, ncore, ncas, x, response_matrix_.data()));
  std::vector<double> kappa_host(mo * mo);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(kappa_host.data(), response_matrix_.data(),
                                         mo * mo * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         stream)));
  FBF_TRY(res.sync());
  std::vector<double> h1(n * n);
  std::vector<double> eri1(n * n * n * n);
  response_integrals(ints_, kappa_host, h1, eri1);
  FBF_TRY(response_sigma_.set_integrals(res, 0.0, h1, eri1));
  FBF_TRY(response_sigma_.apply(res, nroots, ci_c, ci_b_.data()));

  // delta_r: conjugate gradients on (H_0 - E_r) off every root, and the
  // other roots' part exactly, -(c_s . b_r) / (E_s - E_r) c_s.
  double *const cg_r = cg_vectors_.data();
  double *const cg_p = cg_r + dim;
  double *const cg_ap = cg_p + dim;
  for (size_t r = 0; r < roots_count; ++r) {
    const double e_r = roots_->energies[r];
    double *const delta = ci_root(ci_delta_.data(), r);
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(delta, 0, dim * sizeof(double), stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cg_r, ci_root(ci_b_.data(), r), dim * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &kNegOne<double>, cg_r, 1)));
    FBF_TRY(ci_project(cg_r));
    double rhs_norm = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, cg_r, 1, &rhs_norm)));
    double rr = rhs_norm * rhs_norm;
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cg_p, cg_r, dim * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    for (int32_t it = 0; it < kResponseMaxIterations && rr > 0.0; ++it) {
      FBF_TRY(sigma_->apply(res, 1, cg_p, cg_ap));
      const double neg_e = -e_r;
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &neg_e, cg_p, 1, cg_ap, 1)));
      FBF_TRY(ci_project(cg_ap));
      double pap = 0.0;
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, dim32, cg_p, 1, cg_ap, 1, &pap)));
      // H_0 - E_r is positive off the roots in their spin; a direction it
      // is not positive along (roundoff's other spins) ends the solve.
      if (!(pap > 0.0)) {
        break;
      }
      const double alpha_step = rr / pap;
      const double neg_alpha = -alpha_step;
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &alpha_step, cg_p, 1, delta, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &neg_alpha, cg_ap, 1, cg_r, 1)));
      double r_norm = 0.0;
      FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, cg_r, 1, &r_norm)));
      const double rr_new = r_norm * r_norm;
      if (r_norm <= kResponseTolerance * rhs_norm) {
        break;
      }
      const double beta_step = rr_new / rr;
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &beta_step, cg_p, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &kOne<double>, cg_r, 1, cg_p, 1)));
      rr = rr_new;
    }
    for (size_t s = 0; s < roots_count; ++s) {
      const double gap = roots_->energies[s] - e_r;
      if (s == r || std::abs(gap) < kResponseGapFloor) {
        continue;
      }
      double overlap = 0.0;
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, dim32, ci_root(ci_c, s), 1,
                                                   ci_root(ci_b_.data(), r), 1, &overlap)));
      const double coefficient = -overlap / gap;
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, dim32, &coefficient, ci_root(ci_c, s), 1, delta, 1)));
    }
  }

  // gamma_T and Gamma_T: half the difference of the RDMs of c + delta and
  // c - delta (Gamma_T's half again, for the 1/2 of the two-body energy).
  double *const plus = ci_pm_.data();
  double *const minus = plus + ci_block;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                nroots, &kOne<double>, ci_c, dim32, &kOne<double>,
                                                ci_delta_.data(), dim32, plus, dim32)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32, nroots, &kOne<double>, ci_c, dim32,
      &kNegOne<double>, ci_delta_.data(), dim32, minus, dim32)));
  FBF_TRY(rdm_->rdm12(res, weights_, plus, response_gamma_.data(), response_gamma2_.data()));
  FBF_TRY(rdm_->rdm12(res, weights_, minus, response_gamma_.data() + (n * n),
                      response_gamma2_.data() + (n * n * n * n)));
  std::vector<double> g1(2 * n * n);
  std::vector<double> g2(2 * n * n * n * n);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(g1.data(), response_gamma_.data(), g1.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(g2.data(), response_gamma2_.data(), g2.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  for (size_t i = 0; i < gamma_t.size(); ++i) {
    gamma_t[i] = 0.5 * (g1[i] - g1[i + gamma_t.size()]);
  }
  for (size_t i = 0; i < gamma2_t.size(); ++i) {
    gamma2_t[i] = 0.25 * (g2[i] - g2[i + gamma2_t.size()]);
  }
  return {};
}

Status RelaxedHessian::apply(DeviceResources &res, const double *const x, double *const y) {
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(hessian_->apply(res, 1, x, y));

  // R x = H_oc delta, the gradient in x of h_eff'.gamma_T +
  // 1/2 (tu|vw)'.Gamma_T.
  const size_t mo = ints_.nmo;
  const size_t n = ints_.ncas;
  std::vector<double> gamma_t(n * n);
  std::vector<double> gamma2_t(n * n * n * n);
  FBF_TRY(respond(res, x, gamma_t, gamma2_t));
  std::vector<double> a(mo * mo);
  response_gradient(ints_, gamma_t, gamma2_t, a);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(response_matrix_.data(), a.data(), a.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(res.sync());
  FBF_TRY(response::pack_orbital_pairs(res, static_cast<int32_t>(mo),
                                       static_cast<int32_t>(ints_.ncore), static_cast<int32_t>(n),
                                       response_matrix_.data(), response_packed_.data()));
  return gpu_status(wwr::axpy<double, int32_t>(res.blas().get(), n_pairs_, &kOne<double>,
                                               response_packed_.data(), 1, y, 1));
}

} // namespace free_breakfast::mcscf
