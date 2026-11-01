/// @file
/// @brief relaxed_hessian: H'(x) and its adjoint on the host, the CI response
/// by conjugate gradients over the sigma, and H~ x = H x + R x.

module free_breakfast.mcscf.relaxed_hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, geam, dgmm, gemmStridedBatched, axpy, dot, nrm2, ...
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The CI response's conjugate gradients stop after this many sigmas per root
// (or at create's response_tolerance of the right-hand side). Two roots
// closer than kResponseGapFloor (hartree) leave their mixing out.
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

Result<RelaxedHessian> RelaxedHessian::create(
    DeviceResources &res, response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm,
    const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const ci::Roots &roots,
    std::span<const double> weights, ResponseIntegrals ints, const double response_tolerance) {
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
  if (!(response_tolerance > 0.0)) {
    return err_config("mcscf::RelaxedHessian: response_tolerance must be positive (got {:g})",
                      response_tolerance);
  }
  ci::Sigma response_sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  // The roots' parity under the swap of the alpha and beta strings, from
  // c_0 . c_0^T over one set for both spins: +1 for even S, -1 for odd.
  int32_t swap_parity = 0;
  if (alpha.n_electrons() == beta.n_electrons()) {
    const int32_t m = alpha.size();
    Buffers::device<double> swapped(dim, res.device());
    const double *const c0 = roots.vectors.data();
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_T,
                                                  wwr::WWRBLAS_OP_N, m, m, &kOne<double>, c0, m,
                                                  &kZero<double>, c0, m, swapped.data(), m)));
    double overlap = 0.0;
    FBF_TRY(gpu_status(
        wwr::dot<double, int32_t>(res.blas().get(), m * m, c0, 1, swapped.data(), 1, &overlap)));
    swap_parity = overlap > 0.5 ? 1 : (overlap < -0.5 ? -1 : 0);
  }
  return RelaxedHessian{hessian,
                        sigma,
                        rdm,
                        roots,
                        std::vector<double>(weights.begin(), weights.end()),
                        std::move(ints),
                        std::move(response_sigma),
                        dim,
                        response_tolerance,
                        swap_parity,
                        res};
}

RelaxedHessian::RelaxedHessian(response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm,
                               const ci::Roots &roots, std::vector<double> weights,
                               ResponseIntegrals ints, ci::Sigma response_sigma, const size_t dim,
                               const double response_tolerance, const int32_t swap_parity,
                               DeviceResources &res)
    : hessian_(&hessian), sigma_(&sigma), rdm_(&rdm), roots_(&roots), weights_(std::move(weights)),
      ints_(std::move(ints)), n_pairs_(hessian.size()), dim_(dim),
      response_tolerance_(response_tolerance), swap_parity_(swap_parity),
      response_sigma_(std::move(response_sigma)), ci_b_(dim * weights_.size(), res.device()),
      ci_delta_(dim * weights_.size(), res.device()),
      ci_pm_(2 * dim * weights_.size(), res.device()), cg_vectors_(3 * dim, res.device()),
      ci_overlaps_(weights_.size(), res.device()),
      response_matrix_(ints_.nmo * ints_.nmo, res.device()),
      response_packed_(std::max<size_t>(static_cast<size_t>(n_pairs_), 1), res.device()),
      response_gamma_(2 * ints_.ncas * ints_.ncas, res.device()),
      response_gamma2_(2 * ints_.ncas * ints_.ncas * ints_.ncas * ints_.ncas, res.device()),
      block_(1, res.device()) {}

Status RelaxedHessian::project(DeviceResources &res, double *const v) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto dim32 = static_cast<int32_t>(dim_);
  const auto nroots = static_cast<int32_t>(weights_.size());
  const double *const ci_c = roots_->vectors.data();
  FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, dim32, nroots,
                                                &kOne<double>, ci_c, dim32, v, 1, &kZero<double>,
                                                ci_overlaps_.data(), 1)));
  return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, dim32, nroots,
                                               &kNegOne<double>, ci_c, dim32, ci_overlaps_.data(),
                                               1, &kOne<double>, v, 1));
}

Status RelaxedHessian::respond(DeviceResources &res, const double *const x,
                               std::span<double> gamma_t, std::span<double> gamma2_t) {
  if (gamma_t.size() != ints_.ncas * ints_.ncas ||
      gamma2_t.size() != gamma_t.size() * gamma_t.size()) {
    return err_config("mcscf::RelaxedHessian::respond: gamma_t and gamma2_t must be ncas^2 and "
                      "ncas^4 long (ncas {}; got {}, {})",
                      ints_.ncas, gamma_t.size(), gamma2_t.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(prime_sigma(res, x, ci_b_.data()));
  return solve_response(res, gamma_t, gamma2_t);
}

Status RelaxedHessian::prime_sigma(DeviceResources &res, const double *const x, double *const b) {
  const size_t mo = ints_.nmo;
  const size_t n = ints_.ncas;
  const auto nmo = static_cast<int32_t>(mo);
  const auto ncore = static_cast<int32_t>(ints_.ncore);
  const auto ncas = static_cast<int32_t>(n);
  const auto nroots = static_cast<int32_t>(weights_.size());

  // H'(x) from kappa(x), and b_r = H' c_r for every root.
  FBF_TRY(response::unpack_orbital_pairs(res, nmo, ncore, ncas, x, response_matrix_.data()));
  std::vector<double> kappa_host(mo * mo);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(kappa_host.data(), response_matrix_.data(),
                                         mo * mo * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> h1(n * n);
  std::vector<double> eri1(n * n * n * n);
  response_integrals(ints_, kappa_host, h1, eri1);
  FBF_TRY(response_sigma_.set_integrals(res, 0.0, h1, eri1));
  return response_sigma_.apply(res, nroots, roots_->vectors.data(), b);
}

Status RelaxedHessian::couple(DeviceResources &res, const double *const ci_rhs, double *const y) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t n = ints_.ncas;
  const size_t dim = dim_;
  const size_t roots_count = weights_.size();
  // b = y off every root, so that the other roots' part solve_response adds
  // is nothing.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_b_.data(), ci_rhs, dim * roots_count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  for (size_t r = 0; r < roots_count; ++r) {
    FBF_TRY(project(res, ci_b_.data() + (r * dim)));
  }
  std::vector<double> gamma_t(n * n);
  std::vector<double> gamma2_t(n * n * n * n);
  FBF_TRY(solve_response(res, gamma_t, gamma2_t));
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(y, 0, static_cast<size_t>(n_pairs_) * sizeof(double), res.stream())));
  return add_coupling(res, gamma_t, gamma2_t, y);
}

Status RelaxedHessian::solve_response(DeviceResources &res, std::span<double> gamma_t,
                                      std::span<double> gamma2_t) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t dim = dim_;
  const auto dim32 = static_cast<int32_t>(dim);
  const size_t roots_count = weights_.size();
  const double *const ci_c = roots_->vectors.data();
  const auto ci_root = [dim](auto *base, const size_t r) { return base + (r * dim); };

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
    FBF_TRY(project(res, cg_r));
    double rhs_norm = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, cg_r, 1, &rhs_norm)));
    double rr = rhs_norm * rhs_norm;
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cg_p, cg_r, dim * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    for (int32_t it = 0; it < kResponseMaxIterations && rr > 0.0; ++it) {
      FBF_TRY(sigma_->apply(res, 1, cg_p, cg_ap));
      const double neg_e = -e_r;
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &neg_e, cg_p, 1, cg_ap, 1)));
      FBF_TRY(project(res, cg_ap));
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
      if (r_norm <= response_tolerance_ * rhs_norm) {
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

  return transition_rdms(res, ci_delta_.data(), gamma_t, gamma2_t);
}

Status RelaxedHessian::transition_rdms(DeviceResources &res, const double *const delta,
                                       std::span<double> gamma_t, std::span<double> gamma2_t) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t n = ints_.ncas;
  const auto dim32 = static_cast<int32_t>(dim_);
  const auto nroots = static_cast<int32_t>(weights_.size());
  const size_t ci_block = dim_ * weights_.size();
  const double *const ci_c = roots_->vectors.data();

  // gamma_T and Gamma_T: half the difference of the RDMs of c + delta and
  // c - delta (Gamma_T's half again, for the 1/2 of the two-body energy).
  double *const plus = ci_pm_.data();
  double *const minus = plus + ci_block;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                nroots, &kOne<double>, ci_c, dim32, &kOne<double>,
                                                delta, dim32, plus, dim32)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                nroots, &kOne<double>, ci_c, dim32,
                                                &kNegOne<double>, delta, dim32, minus, dim32)));
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

Status RelaxedHessian::add_coupling(DeviceResources &res, std::span<const double> gamma_t,
                                    std::span<const double> gamma2_t, double *const y) {
  const size_t mo = ints_.nmo;
  std::vector<double> a(mo * mo);
  response_gradient(ints_, gamma_t, gamma2_t, a);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(response_matrix_.data(), a.data(), a.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(res.sync());
  FBF_TRY(response::pack_orbital_pairs(
      res, static_cast<int32_t>(mo), static_cast<int32_t>(ints_.ncore),
      static_cast<int32_t>(ints_.ncas), response_matrix_.data(), response_packed_.data()));
  return gpu_status(wwr::axpy<double, int32_t>(res.blas().get(), n_pairs_, &kOne<double>,
                                               response_packed_.data(), 1, y, 1));
}

Status RelaxedHessian::apply(DeviceResources &res, const double *const x, double *const y) {
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(hessian_->apply(res, 1, x, y));

  // R x = H_oc delta, the gradient in x of h_eff'.gamma_T +
  // 1/2 (tu|vw)'.Gamma_T.
  const size_t n = ints_.ncas;
  std::vector<double> gamma_t(n * n);
  std::vector<double> gamma2_t(n * n * n * n);
  FBF_TRY(respond(res, x, gamma_t, gamma2_t));
  return add_coupling(res, gamma_t, gamma2_t, y);
}

void RelaxedHessian::reserve(DeviceResources &res, const size_t k) {
  const size_t ci_block = dim_ * weights_.size();
  const size_t count = k * weights_.size();
  if (ci_b_.num_elements() < k * ci_block) {
    ci_b_ = Buffers::device<double>(k * ci_block, res.device());
  }
  if (ci_delta_.num_elements() < k * ci_block) {
    ci_delta_ = Buffers::device<double>(k * ci_block, res.device());
  }
  // apply_coupled's rotation matrices (nroots x nroots per column) after the
  // overlaps.
  const size_t lockstep = (4 * dim_ * count) + (4 * count) + (2 * weights_.size() * count);
  if (block_.num_elements() < lockstep) {
    block_ = Buffers::device<double>(lockstep, res.device());
  }
}

Status RelaxedHessian::project_block(DeviceResources &res, const size_t count, double *const v) {
  const auto dim32 = static_cast<int32_t>(dim_);
  const auto nroots = static_cast<int32_t>(weights_.size());
  const auto n = static_cast<int32_t>(count);
  // The overlaps sit past the lockstep vectors and scalars (reserve's
  // layout), which nothing else holds across this call.
  double *const overlaps = block_.data() + (4 * dim_ * count) + (4 * count);
  const double *const ci_c = roots_->vectors.data();
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas().get(), nroots, n, dim32, &kOne<double>,
                                             ci_c, v, &kZero<double>, overlaps)));
  return gpu_status(
      gemm(res.blas().get(), dim32, n, nroots, &kNegOne<double>, ci_c, overlaps, &kOne<double>, v));
}

Status RelaxedHessian::shifted_sigma(DeviceResources &res, const size_t k, const double *const x_c,
                                     double *const y_c) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const size_t roots_count = weights_.size();
  const size_t count = k * roots_count;
  const auto dim32 = static_cast<int32_t>(dim_);
  const auto n = static_cast<int32_t>(count);
  double *const product = block_.data() + (3 * dim_ * count);
  double *const shifts = block_.data() + (4 * dim_ * count);
  std::vector<double> e(count);
  for (size_t v = 0; v < count; ++v) {
    e[v] = roots_->energies[v % roots_count];
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(shifts, e.data(), count * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(sigma_->apply(res, n, x_c, y_c));
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim32, n, x_c, dim32,
                                                shifts, 1, product, dim32)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                n, &kOne<double>, y_c, dim32, &kNegOne<double>,
                                                product, dim32, y_c, dim32)));
  // The host shifts go out of scope: the copy must have left them.
  return res.sync();
}

Status RelaxedHessian::solve_responses(DeviceResources &res, const size_t k) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t roots_count = weights_.size();
  const size_t count = k * roots_count;
  const size_t dim = dim_;
  const auto dim32 = static_cast<int32_t>(dim);
  const auto n = static_cast<int32_t>(count);
  const auto nroots = static_cast<int32_t>(roots_count);
  const size_t block = dim * count;
  double *const delta = ci_delta_.data();
  const double *const b = ci_b_.data();
  // reserve's layout: r, P, A P, a product, then the shifts (shifted_sigma's),
  // the dots and the step lengths; the overlaps (project_block's) after.
  double *const r = block_.data();
  double *const p = r + block;
  double *const ap = p + block;
  double *const product = ap + block;
  double *const dots = product + block + count;
  double *const steps = dots + (2 * count);
  double *const overlaps = steps + count;

  // Each vector's dot with its own partner, read back.
  std::vector<double> host(count);
  const auto column_dots = [&](const double *u, const double *w) -> Status {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, 1, 1, dim32, &kOne<double>, u, dim32, dim32, w,
        dim32, dim32, &kZero<double>, dots, 1, 1, n)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), dots, count * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    return res.sync();
  };
  // a <- a + sign diag(steps) b over the vectors, steps from the host.
  std::vector<double> step_host(count);
  const auto scaled_add = [&](double *a, const double *u, const double *sign) -> Status {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(steps, step_host.data(), count * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim32, n, u, dim32,
                                                  steps, 1, product, dim32)));
    return gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                 n, &kOne<double>, a, dim32, sign, product, dim32,
                                                 a, dim32));
  };

  // delta = 0, r = -P b, p = r: conjugate gradients on (H_0 - E_r) off every
  // root, each vector at its own root's energy, in lockstep.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(delta, 0, block * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                n, &kNegOne<double>, b, dim32, &kZero<double>, b,
                                                dim32, r, dim32)));
  FBF_TRY(project_block(res, count, r));
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p, r, block * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(column_dots(r, r));
  std::vector<double> rr = host;
  std::vector<double> rhs_norm(count);
  std::vector<bool> active(count);
  for (size_t v = 0; v < count; ++v) {
    rhs_norm[v] = std::sqrt(rr[v]);
    active[v] = rr[v] > 0.0;
  }
  for (int32_t it = 0; it < kResponseMaxIterations && std::ranges::contains(active, true); ++it) {
    FBF_TRY(shifted_sigma(res, k, p, ap));
    FBF_TRY(project_block(res, count, ap));
    FBF_TRY(column_dots(p, ap));
    // H_0 - E_r is positive off the roots in their spin; a direction it is
    // not positive along (roundoff's other spins) ends that vector's solve.
    for (size_t v = 0; v < count; ++v) {
      active[v] = active[v] && host[v] > 0.0;
      step_host[v] = active[v] ? rr[v] / host[v] : 0.0;
    }
    FBF_TRY(scaled_add(delta, p, &kOne<double>));
    FBF_TRY(scaled_add(r, ap, &kNegOne<double>));
    FBF_TRY(column_dots(r, r));
    for (size_t v = 0; v < count; ++v) {
      if (active[v] && std::sqrt(host[v]) <= response_tolerance_ * rhs_norm[v]) {
        active[v] = false;
      }
      step_host[v] = active[v] ? host[v] / rr[v] : 0.0;
      rr[v] = active[v] ? host[v] : rr[v];
    }
    // p <- r + beta p.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(steps, step_host.data(), count * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim32, n, p, dim32,
                                                  steps, 1, product, dim32)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32,
                                                  n, &kOne<double>, r, dim32, &kOne<double>,
                                                  product, dim32, p, dim32)));
  }

  // The other roots' part exactly, -(c_s . b_r) / (E_s - E_r) c_s, for every
  // vector at once: the overlaps C^T b read back, scaled on the host and
  // C times them added.
  const double *const ci_c = roots_->vectors.data();
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nroots, n, dim32, &kOne<double>, ci_c, b,
                                             &kZero<double>, overlaps)));
  std::vector<double> coefficients(roots_count * count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(coefficients.data(), overlaps,
                                         coefficients.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  for (size_t v = 0; v < count; ++v) {
    const double e_r = roots_->energies[v % roots_count];
    for (size_t s = 0; s < roots_count; ++s) {
      const double gap = roots_->energies[s] - e_r;
      double &value = coefficients[s + (roots_count * v)];
      value = (s == v % roots_count || std::abs(gap) < kResponseGapFloor) ? 0.0 : -value / gap;
    }
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(overlaps, coefficients.data(),
                                         coefficients.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(
      gemm(blas, dim32, n, nroots, &kOne<double>, ci_c, overlaps, &kOne<double>, delta)));
  return res.sync();
}

Status RelaxedHessian::add_couplings(DeviceResources &res, const size_t k,
                                     const double *const delta, double *const y) {
  const size_t n = ints_.ncas;
  const size_t ci_block = dim_ * weights_.size();
  const auto pairs = static_cast<size_t>(n_pairs_);
  std::vector<double> gamma_t(n * n);
  std::vector<double> gamma2_t(n * n * n * n);
  for (size_t j = 0; j < k; ++j) {
    FBF_TRY(transition_rdms(res, delta + (j * ci_block), gamma_t, gamma2_t));
    FBF_TRY(add_coupling(res, gamma_t, gamma2_t, y + (j * pairs)));
  }
  return {};
}

Status RelaxedHessian::apply(DeviceResources &res, const int32_t k, const double *const x,
                             double *const y) {
  if (k < 0) {
    return err_config("mcscf::RelaxedHessian::apply: the block has {} columns", k);
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto columns = static_cast<size_t>(k);
  const size_t ci_block = dim_ * weights_.size();
  const auto pairs = static_cast<size_t>(n_pairs_);
  reserve(res, columns);
  FBF_TRY(hessian_->apply(res, k, x, y));
  for (size_t j = 0; j < columns; ++j) {
    FBF_TRY(prime_sigma(res, x + (j * pairs), ci_b_.data() + (j * ci_block)));
  }
  FBF_TRY(solve_responses(res, columns));
  return add_couplings(res, columns, ci_delta_.data(), y);
}

Status RelaxedHessian::couple(DeviceResources &res, const int32_t k, const double *const ci_rhs,
                              double *const y) {
  if (k < 0) {
    return err_config("mcscf::RelaxedHessian::couple: the block has {} columns", k);
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto columns = static_cast<size_t>(k);
  const size_t ci_block = dim_ * weights_.size();
  reserve(res, columns);
  // b = y off every root, so that the other roots' part is nothing.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_b_.data(), ci_rhs, columns * ci_block * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(project_block(res, columns * weights_.size(), ci_b_.data()));
  FBF_TRY(solve_responses(res, columns));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
      y, 0, columns * static_cast<size_t>(n_pairs_) * sizeof(double), res.stream())));
  return add_couplings(res, columns, ci_delta_.data(), y);
}

Status RelaxedHessian::apply_coupled(DeviceResources &res, const int32_t k, const double *const x,
                                     const double *const x_c, double *const y, double *const y_c,
                                     std::span<const double> theta, std::span<double> y_theta) {
  const size_t roots_count = weights_.size();
  const size_t n_theta = roots_count * (roots_count - 1) / 2;
  if (k < 0 || (!theta.empty() && (theta.size() != static_cast<size_t>(k) * n_theta ||
                                   y_theta.size() != theta.size()))) {
    return err_config("mcscf::RelaxedHessian::apply_coupled: the block has {} columns, theta {} "
                      "and y_theta {} doubles (none, or {} rotations a column)",
                      k, theta.size(), y_theta.size(), n_theta);
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto columns = static_cast<size_t>(k);
  const size_t ci_block = dim_ * roots_count;
  const size_t count = columns * roots_count;
  const auto pairs = static_cast<size_t>(n_pairs_);
  const auto dim32 = static_cast<int32_t>(dim_);
  const auto nroots = static_cast<int32_t>(roots_count);
  reserve(res, columns);
  // reserve's layout: the lockstep r is free here, and the rotation matrices
  // sit after the overlaps.
  double *const overlaps = block_.data() + (4 * dim_ * count) + (4 * count);
  double *const rotations = overlaps + (count * roots_count);
  // With rotations, x_c gains their part along the roots, x_c,r +=
  // sum_s Theta_sr c_s: one gemm of C by the k antisymmetric matrices, into
  // the lockstep r, which then stands in for x_c.
  const double *vectors = x_c;
  if (!theta.empty()) {
    std::vector<double> m(count * roots_count, 0.0);
    for (size_t j = 0; j < columns; ++j) {
      size_t at = j * n_theta;
      for (size_t r = 0; r < roots_count; ++r) {
        for (size_t s = r + 1; s < roots_count; ++s, ++at) {
          m[s + (roots_count * (r + (roots_count * j)))] = theta[at];
          m[r + (roots_count * (s + (roots_count * j)))] = -theta[at];
        }
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rotations, m.data(), m.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    double *const shifted = block_.data();
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(shifted, x_c, columns * ci_block * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(gpu_status(gemm(blas, dim32, static_cast<int32_t>(count), nroots, &kOne<double>,
                            roots_->vectors.data(), rotations, &kOne<double>, shifted)));
    // The host matrices go out of scope: the copy must have left them.
    FBF_TRY(res.sync());
    vectors = shifted;
  }
  FBF_TRY(hessian_->apply(res, k, x, y));
  // y_c = (H_0 - E_r) x_c, then H'(x_j) c added per column through ci_b_,
  // and the sum projected.
  FBF_TRY(shifted_sigma(res, columns, vectors, y_c));
  for (size_t j = 0; j < columns; ++j) {
    FBF_TRY(prime_sigma(res, x + (j * pairs), ci_b_.data() + (j * ci_block)));
  }
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(res.blas().get(),
                                                static_cast<int32_t>(columns * ci_block),
                                                &kOne<double>, ci_b_.data(), 1, y_c, 1)));
  FBF_TRY(project_block(res, count, y_c));
  // The rotations' rows: y_c's overlaps with the roots before the projection
  // (project_block leaves them), Y_j[s, r] = c_s . (H'(x_j) c_r +
  // (H_0 - E_r) x_c,r), both orders averaged.
  if (!theta.empty()) {
    std::vector<double> o(count * roots_count);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(o.data(), overlaps, o.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    for (size_t j = 0; j < columns; ++j) {
      size_t at = j * n_theta;
      const double *const oj = o.data() + (j * roots_count * roots_count);
      for (size_t r = 0; r < roots_count; ++r) {
        for (size_t s = r + 1; s < roots_count; ++s, ++at) {
          y_theta[at] = 0.5 * (oj[s + (roots_count * r)] + oj[r + (roots_count * s)]);
        }
      }
    }
  }
  // y_c into the roots' sector under the swap of the strings, (y_c + p y_c^T)
  // / 2 for their parity p: exactly, so that iterates built from it by axpy
  // and a swap-symmetric diagonal stay there too. The projection's roundoff
  // otherwise seeds the other sector, which for an excited root can hold a
  // state of another spin below it.
  if (swap_parity_ != 0) {
    const auto m = static_cast<int32_t>(std::lround(std::sqrt(static_cast<double>(dim_))));
    const double half_swapped = 0.5 * swap_parity_;
    double *const product = block_.data() + (3 * dim_ * count);
    for (size_t v = 0; v < count; ++v) {
      const double *const from = y_c + (v * dim_);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m, m, &kHalf<double>, from, m,
          &half_swapped, from, m, product + (v * dim_), m)));
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(y_c, product, count * dim_ * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  return add_couplings(res, columns, vectors, y);
}

Status RelaxedHessian::coupling(DeviceResources &res, const int32_t k, const double *const x_c,
                                double *const y) {
  if (k < 0) {
    return err_config("mcscf::RelaxedHessian::coupling: the block has {} columns", k);
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto columns = static_cast<size_t>(k);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
      y, 0, columns * static_cast<size_t>(n_pairs_) * sizeof(double), res.stream())));
  return add_couplings(res, columns, x_c, y);
}

Status RelaxedHessian::ci_diagonal(DeviceResources &res, double *const diagonal) {
  const ScopedDeviceIndex on{res.dev_idx()};
  return sigma_->diagonal(res, diagonal);
}

} // namespace free_breakfast::mcscf
