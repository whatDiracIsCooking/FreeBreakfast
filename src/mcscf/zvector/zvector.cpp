/// @file
/// @brief zvector: the CI right-hand side's response folded into the orbital
/// one, preconditioned conjugate gradients over H~ for z_o, and z_c from the
/// CI response at the solution plus the right-hand side's.

module free_breakfast.mcscf.zvector;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, dgmm, axpy, dot, nrm2, scal, gpu_status
import free_breakfast.core.resources;
import free_breakfast.ci.solver;
import free_breakfast.response.hessian;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// A pair whose diag(H) is below this is left unscaled by the preconditioner.
// At a minimum diag(H) >= diag(H~) > 0, so it guards only roundoff.
constexpr double kZvectorDiagonalFloor = 1e-8;
// The coupled solver's CI preconditioner is 1 / (2 w_r |diag(H_0) - E_r|),
// the difference raised to this where it is smaller: root r's own leading
// determinants sit near E_r, and a pole there would amplify the directions
// the projection removes.
constexpr double kCoupledCiDiagonalFloor = 1e-2;

// diag(H)^-1 into the size() device doubles @p inverse, floored, built on
// the host.
Status inverse_orbital_diagonal(DeviceResources &res, const response::Hessian &hessian,
                                double *const inverse) {
  const auto pairs = static_cast<size_t>(hessian.size());
  std::vector<double> scale(pairs);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scale.data(), hessian.diagonal().data(),
                                         pairs * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         res.stream())));
  FBF_TRY(res.sync());
  for (double &d : scale) {
    d = d > kZvectorDiagonalFloor ? 1.0 / d : 1.0;
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(inverse, scale.data(), pairs * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  return res.sync();
}

// Two weights this close, relative to the larger, are equal: the rotation
// between their roots is redundant and stays projected. Weights recovered by
// a least-squares solve (mcscf::df_gradient's) are equal only to about this.
constexpr double kRotationWeightTolerance = 1e-8;
// Two roots closer than this (hartree) are not rotated by a right-hand side,
// as RelaxedHessian leaves their mixing out of the CI response.
constexpr double kRotationGapFloor = 1e-8;

// Which of the nroots (nroots - 1) / 2 pairs r < s, in (r, s) lexicographic
// order, rotate: those at unequal weights.
std::vector<bool> rotating_pairs(const std::span<const double> weights) {
  const size_t nroots = weights.size();
  std::vector<bool> rotating;
  rotating.reserve(nroots * (nroots - 1) / 2);
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots; ++s) {
      const double scale = std::max(std::abs(weights[r]), std::abs(weights[s]));
      rotating.push_back(std::abs(weights[r] - weights[s]) > kRotationWeightTolerance * scale);
    }
  }
  return rotating;
}

// The k columns' rotations @p theta (host, packed) added to their CI blocks
// @p ci on the device along the roots: ci_j,r += sum_s Theta_sr c_s, one gemm
// of C by the antisymmetric matrices.
Status add_rotations(DeviceResources &res, const ci::Roots &roots, const size_t k,
                     const std::span<const double> theta, double *const ci) {
  const size_t nroots = roots.energies.size();
  const size_t dim = roots.vectors.num_elements() / nroots;
  std::vector<double> m(nroots * nroots * k, 0.0);
  size_t at = 0;
  for (size_t j = 0; j < k; ++j) {
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = r + 1; s < nroots; ++s, ++at) {
        m[s + (nroots * (r + (nroots * j)))] = theta[at];
        m[r + (nroots * (s + (nroots * j)))] = -theta[at];
      }
    }
  }
  Buffers::device<double> device(m.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(device.data(), m.data(), m.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(gemm(res.blas().get(), static_cast<int32_t>(dim),
                          static_cast<int32_t>(nroots * k), static_cast<int32_t>(nroots),
                          &kOne<double>, roots.vectors.data(), device.data(), &kOne<double>, ci)));
  return res.sync();
}

// Theta_sr of the k CI blocks @p ci on the device, 1/2 (c_s . ci_r -
// c_r . ci_s) for the rotating pairs and zero for the rest, packed.
Result<std::vector<double>> read_rotations(DeviceResources &res, const ci::Roots &roots,
                                           const std::vector<bool> &rotating, const size_t k,
                                           const double *const ci) {
  const size_t nroots = roots.energies.size();
  const size_t dim = roots.vectors.num_elements() / nroots;
  Buffers::device<double> overlaps(nroots * nroots * k, res.device());
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(
      res.blas().get(), static_cast<int32_t>(nroots), static_cast<int32_t>(nroots * k),
      static_cast<int32_t>(dim), &kOne<double>, roots.vectors.data(), ci, &kZero<double>,
      overlaps.data())));
  std::vector<double> o(nroots * nroots * k);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(o.data(), overlaps.data(), o.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> theta(rotating.size() * k, 0.0);
  size_t at = 0;
  for (size_t j = 0; j < k; ++j) {
    const double *const oj = o.data() + (j * nroots * nroots);
    size_t pair = 0;
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = r + 1; s < nroots; ++s, ++at, ++pair) {
        if (rotating[pair]) {
          theta[at] = 0.5 * (oj[s + (nroots * r)] - oj[r + (nroots * s)]);
        }
      }
    }
  }
  return theta;
}

// A rotation right-hand side t (k columns, packed) folded in as couple folds
// y: Theta_w,sr = -t_sr / (E_s - E_r) for the rotating pairs, its part along
// the roots added to the k CI blocks @p ci (z_c's start), and H_oc of it to
// the size() x k orbital right-hand sides @p b, through @p scratch (as b).
Status couple_rotations(DeviceResources &res, RelaxedHessian &relaxed, const ci::Roots &roots,
                        const std::vector<bool> &rotating, const size_t k,
                        const std::span<const double> theta_rhs, double *const b, double *const ci,
                        double *const scratch) {
  const size_t nroots = roots.energies.size();
  std::vector<double> theta(theta_rhs.size(), 0.0);
  size_t at = 0;
  for (size_t j = 0; j < k; ++j) {
    size_t pair = 0;
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = r + 1; s < nroots; ++s, ++at, ++pair) {
        const double gap = roots.energies[s] - roots.energies[r];
        if (rotating[pair] && std::abs(gap) >= kRotationGapFloor) {
          theta[at] = -theta_rhs[at] / gap;
        }
      }
    }
  }
  const size_t ci_block = roots.vectors.num_elements();
  Buffers::device<double> along(ci_block * k, res.device());
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(along.data(), 0, ci_block * k * sizeof(double), res.stream())));
  FBF_TRY(add_rotations(res, roots, k, theta, along.data()));
  FBF_TRY(relaxed.coupling(res, static_cast<int32_t>(k), along.data(), scratch));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, relaxed.size() * static_cast<int32_t>(k),
                                                &kOne<double>, scratch, 1, b, 1)));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(ci_block * k),
                                                &kOne<double>, along.data(), 1, ci, 1)));
  return res.sync();
}

Result<ZVector> zvector(DeviceResources &res, RelaxedHessian &relaxed,
                        const response::Hessian &hessian, const ci::Roots &roots,
                        const double *const gradient, const double *const ci_rhs,
                        const double tolerance, const int32_t max_iterations,
                        const std::span<const double> theta_rhs) {
  const int32_t n_pairs = relaxed.size();
  const size_t roots_count = roots.energies.size();
  if (hessian.size() != n_pairs || roots_count == 0 ||
      roots.vectors.num_elements() % roots_count != 0 || !(tolerance > 0.0) || max_iterations < 1 ||
      (!theta_rhs.empty() && theta_rhs.size() != roots_count * (roots_count - 1) / 2)) {
    return err_config("mcscf::zvector: needs H of H~'s {} pairs, at least one root as a whole "
                      "number of vectors, tolerance > 0, max_iterations >= 1 and no rotation "
                      "right-hand side or one per pair of roots (H {}, {} roots in {} doubles, "
                      "tolerance {:g}, max_iterations {}, {} rotations)",
                      n_pairs, hessian.size(), roots_count, roots.vectors.num_elements(), tolerance,
                      max_iterations, theta_rhs.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto pairs = static_cast<size_t>(n_pairs);
  const size_t dim = roots.vectors.num_elements() / roots_count;
  const auto dim32 = static_cast<int32_t>(dim);
  const auto nroots = static_cast<int32_t>(roots_count);

  ZVector out{Buffers::device<double>(std::max<size_t>(pairs, 1), res.device()),
              Buffers::device<double>(dim * roots_count, res.device()),
              {},
              0,
              0.0};
  const std::vector<bool> rotating = rotating_pairs(relaxed.weights());
  const bool rotates = std::ranges::contains(rotating, true);
  // The residual r, the direction p, H~ p, the preconditioned residual,
  // diag(H)^-1 and the orbital right-hand side b = g_I + H_oc w: held across
  // H~'s products, which take the workspace, so its own (docs/style.md
  // "Scratch memory"). And C^T delta_r's nroots overlaps.
  Buffers::device<double> pcg(std::max<size_t>(6 * pairs, 1), res.device());
  Buffers::device<double> overlaps(roots_count, res.device());
  double *const z = out.orbital.data();
  double *const r = pcg.data();
  double *const p = r + pairs;
  double *const hp = p + pairs;
  double *const s = hp + pairs;
  double *const inverse_diagonal = s + pairs;
  double *const b = inverse_diagonal + pairs;

  // b = g_I + H_oc w for w_r = -(H_0 - E_r)^-1 P y_r, which z_c keeps
  // (out.ci) until z_o's response is added to it.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(b, gradient, pairs * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  if (ci_rhs == nullptr) {
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(out.ci.data(), 0, dim * roots_count * sizeof(double), stream)));
  } else {
    FBF_TRY(relaxed.couple(res, ci_rhs, hp));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>, hp, 1, b, 1)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.ci.data(), relaxed.delta(),
                                           dim * roots_count * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  if (rotates && !theta_rhs.empty()) {
    FBF_TRY(couple_rotations(res, relaxed, roots, rotating, 1, theta_rhs, b, out.ci.data(), hp));
  }

  FBF_TRY(inverse_orbital_diagonal(res, hessian, inverse_diagonal));
  // s <- diag(H)^-1 r, elementwise: each element reads only its own.
  const auto precondition = [&]() -> Status {
    return gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_pairs, 1, r,
                                                 n_pairs, inverse_diagonal, 1, s, n_pairs));
  };
  const auto dot = [&](const double *a, const double *b) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n_pairs, a, 1, b, 1, &value)));
    return value;
  };

  // z = 0, r = -b, p = s = M r.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(z, 0, pairs * sizeof(double), stream)));
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(r, b, pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n_pairs, &kNegOne<double>, r, 1)));
  FBF_TRY(precondition());
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p, s, pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  double rs = FBF_TRY(dot(r, s));
  double r_norm = 0.0;
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n_pairs, r, 1, &r_norm)));
  while (r_norm > tolerance) {
    if (out.iterations == max_iterations) {
      return err_max_iterations_exceeded(
          "mcscf::zvector: ||r|| {:.3e} after {} products, above {:.3e}", r_norm, max_iterations,
          tolerance);
    }
    FBF_TRY(relaxed.apply(res, p, hp));
    ++out.iterations;
    const double php = FBF_TRY(dot(p, hp));
    if (!(php > 0.0)) {
      return err_divergence("mcscf::zvector: p.H~p = {:.3e} after {} products: H~ is not positive "
                            "definite, so the orbitals are not a minimum",
                            php, out.iterations);
    }
    const double alpha = rs / php;
    const double neg_alpha = -alpha;
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &alpha, p, 1, z, 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &neg_alpha, hp, 1, r, 1)));
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n_pairs, r, 1, &r_norm)));
    FBF_TRY(precondition());
    const double rs_new = FBF_TRY(dot(r, s));
    const double beta = rs_new / rs;
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n_pairs, &beta, p, 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>, s, 1, p, 1)));
    rs = rs_new;
  }

  // The true residual H~ z + b, from one more product: the recurrence's r
  // drifts from it, H~'s products being good only to the CI response's
  // conjugate gradients. That product's respond leaves delta(z_o) in
  // relaxed.delta(), which with w is z_c once off every root (the other
  // roots' part respond adds is the state average's redundant rotations).
  FBF_TRY(relaxed.apply(res, z, hp));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>, b, 1, hp, 1)));
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n_pairs, hp, 1, &out.residual)));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32 * nroots, &kOne<double>,
                                                relaxed.delta(), 1, out.ci.data(), 1)));
  // The rotations are z_c's part along the other roots, for the pairs of
  // unequal weights (the rest are redundant): read before it is projected.
  out.theta = rotates ? FBF_TRY(read_rotations(res, roots, rotating, 1, out.ci.data()))
                      : std::vector<double>(rotating.size(), 0.0);
  const double *const c = roots.vectors.data();
  for (size_t root = 0; root < roots_count; ++root) {
    double *const v = out.ci.data() + (root * dim);
    FBF_TRY(
        gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, dim32, nroots, &kOne<double>,
                                              c, dim32, v, 1, &kZero<double>, overlaps.data(), 1)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, dim32, nroots,
                                                  &kNegOne<double>, c, dim32, overlaps.data(), 1,
                                                  &kOne<double>, v, 1)));
  }
  FBF_TRY(res.sync());
  return out;
}

// The count vectors of @p v (dim x count) off every root: v <- v - C C^T v,
// through the nroots x count @p overlaps.
Status project_off_roots(DeviceResources &res, const ci::Roots &roots, const size_t dim,
                         const size_t count, double *const v, double *const overlaps) {
  const auto dim32 = static_cast<int32_t>(dim);
  const auto nroots = static_cast<int32_t>(roots.energies.size());
  const auto n = static_cast<int32_t>(count);
  const double *const c = roots.vectors.data();
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas().get(), nroots, n, dim32, &kOne<double>, c,
                                             v, &kZero<double>, overlaps)));
  return gpu_status(
      gemm(res.blas().get(), dim32, n, nroots, &kNegOne<double>, c, overlaps, &kOne<double>, v));
}

// The columns still above the tolerance, or MaxIterationsExceeded once every
// sweep is spent with one left.
Result<std::vector<size_t>> active_columns(std::span<const double> r_norm, const double tolerance,
                                           const int32_t sweeps, const int32_t max_iterations,
                                           std::string_view method) {
  std::vector<size_t> active;
  for (size_t x = 0; x < r_norm.size(); ++x) {
    if (r_norm[x] > tolerance) {
      active.push_back(x);
    }
  }
  if (!active.empty() && sweeps == max_iterations) {
    return err_max_iterations_exceeded(
        "mcscf::zvectors ({}): {} of {} columns above {:.3e} after {} products, the largest "
        "||r|| {:.3e}",
        method, active.size(), r_norm.size(), tolerance, max_iterations, std::ranges::max(r_norm));
  }
  return active;
}

// zvector on each column in turn.
Result<ZVectors> zvectors_per_column(DeviceResources &res, RelaxedHessian &relaxed,
                                     const response::Hessian &hessian, const ci::Roots &roots,
                                     const size_t ncol, const double *const gradients,
                                     const double *const ci_rhs, const double tolerance,
                                     const int32_t max_iterations,
                                     const std::span<const double> theta_rhs, ZVectors &out) {
  const auto pairs = static_cast<size_t>(relaxed.size());
  const size_t ci_block = roots.vectors.num_elements();
  const size_t n_theta = out.theta.size() / ncol;
  for (size_t x = 0; x < ncol; ++x) {
    const ZVector z = FBF_TRY(
        zvector(res, relaxed, hessian, roots, gradients + (x * pairs),
                ci_rhs == nullptr ? nullptr : ci_rhs + (x * ci_block), tolerance, max_iterations,
                theta_rhs.empty() ? theta_rhs : theta_rhs.subspan(x * n_theta, n_theta)));
    std::ranges::copy(z.theta, out.theta.begin() + static_cast<std::ptrdiff_t>(x * n_theta));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.orbital.data() + (x * pairs), z.orbital.data(),
                                           pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                           res.stream())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.ci.data() + (x * ci_block), z.ci.data(),
                                           ci_block * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                           res.stream())));
    // z's buffers are freed at the end of the iteration: the copies first.
    FBF_TRY(res.sync());
    out.iterations[x] = z.iterations;
    out.residuals[x] = z.residual;
  }
  return std::move(out);
}

// The Schur complement in lockstep: zvector's conjugate gradients on every
// column, each sweep one block product of H~ over the active columns.
Result<ZVectors> zvectors_schur(DeviceResources &res, RelaxedHessian &relaxed,
                                const response::Hessian &hessian, const ci::Roots &roots,
                                const size_t ncol, const double *const gradients,
                                const double *const ci_rhs, const double tolerance,
                                const int32_t max_iterations,
                                const std::span<const double> theta_rhs, ZVectors &out) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t n_pairs = relaxed.size();
  const auto pairs = static_cast<size_t>(n_pairs);
  const size_t ci_block = roots.vectors.num_elements();
  const size_t roots_count = roots.energies.size();
  const size_t dim = ci_block / roots_count;
  const size_t block = pairs * ncol;
  const auto ncol32 = static_cast<int32_t>(ncol);
  const std::vector<bool> rotating = rotating_pairs(relaxed.weights());
  const bool rotates = std::ranges::contains(rotating, true);

  // R, P, H~ P (scattered), S, B = g + H_oc w, the gathered P and H~ P, and
  // diag(H)^-1: held across H~'s products, which take the workspace, so its
  // own. And the roots' overlaps for z_c's projection.
  Buffers::device<double> pcg(std::max<size_t>((7 * block) + pairs, 1), res.device());
  Buffers::device<double> overlaps(roots_count * roots_count * ncol, res.device());
  double *const z = out.orbital.data();
  double *const r = pcg.data();
  double *const p = r + block;
  double *const hp = p + block;
  double *const s = hp + block;
  double *const b = s + block;
  double *const xs = b + block;
  double *const ys = xs + block;
  double *const inverse_diagonal = ys + block;

  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(b, gradients, block * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  if (ci_rhs == nullptr) {
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(out.ci.data(), 0, ci_block * ncol * sizeof(double), stream)));
  } else {
    FBF_TRY(relaxed.couple(res, ncol32, ci_rhs, hp));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(block), &kOne<double>, hp, 1, b, 1)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.ci.data(), relaxed.delta(),
                                           ci_block * ncol * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  if (rotates && !theta_rhs.empty()) {
    FBF_TRY(couple_rotations(res, relaxed, roots, rotating, ncol, theta_rhs, b, out.ci.data(), hp));
  }
  FBF_TRY(inverse_orbital_diagonal(res, hessian, inverse_diagonal));
  const auto precondition = [&]() -> Status {
    return gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_pairs, ncol32, r,
                                                 n_pairs, inverse_diagonal, 1, s, n_pairs));
  };
  const auto dot = [&](const double *u, const double *v) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n_pairs, u, 1, v, 1, &value)));
    return value;
  };
  const auto norm = [&](const double *u) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n_pairs, u, 1, &value)));
    return value;
  };

  // Z = 0, R = -B, P = S = M R.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(z, 0, block * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_pairs,
                                                ncol32, &kNegOne<double>, b, n_pairs,
                                                &kZero<double>, b, n_pairs, r, n_pairs)));
  FBF_TRY(precondition());
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p, s, block * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  std::vector<double> rs(ncol);
  std::vector<double> r_norm(ncol);
  for (size_t x = 0; x < ncol; ++x) {
    rs[x] = FBF_TRY(dot(r + (x * pairs), s + (x * pairs)));
    r_norm[x] = FBF_TRY(norm(r + (x * pairs)));
  }

  for (int32_t sweeps = 0;; ++sweeps) {
    const std::vector<size_t> active =
        FBF_TRY(active_columns(r_norm, tolerance, sweeps, max_iterations, "Schur"));
    if (active.empty()) {
      break;
    }
    // H~ P over the active columns, gathered into XS and back from YS.
    for (size_t i = 0; i < active.size(); ++i) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(xs + (i * pairs), p + (active[i] * pairs),
                                             pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                             stream)));
    }
    FBF_TRY(relaxed.apply(res, static_cast<int32_t>(active.size()), xs, ys));
    for (size_t i = 0; i < active.size(); ++i) {
      const size_t x = active[i];
      const size_t at = x * pairs;
      const double *const hpx = ys + (i * pairs);
      ++out.iterations[x];
      const double php = FBF_TRY(dot(p + at, hpx));
      if (!(php > 0.0)) {
        return err_divergence("mcscf::zvectors (Schur): p.H~p = {:.3e} in column {} after {} "
                              "products: H~ is not positive definite, so the orbitals are not a "
                              "minimum",
                              php, x, out.iterations[x]);
      }
      const double alpha = rs[x] / php;
      const double neg_alpha = -alpha;
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &alpha, p + at, 1, z + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &neg_alpha, hpx, 1, r + at, 1)));
      r_norm[x] = FBF_TRY(norm(r + at));
    }
    FBF_TRY(precondition());
    for (const size_t x : active) {
      const size_t at = x * pairs;
      const double rs_new = FBF_TRY(dot(r + at, s + at));
      const double beta = rs_new / rs[x];
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n_pairs, &beta, p + at, 1)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>, s + at, 1, p + at, 1)));
      rs[x] = rs_new;
    }
  }

  // The true residuals H~ Z + B from one more block product, whose deltas
  // with w are z_c once off every root (zvector's step 3).
  FBF_TRY(relaxed.apply(res, ncol32, z, hp));
  FBF_TRY(gpu_status(
      wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(block), &kOne<double>, b, 1, hp, 1)));
  for (size_t x = 0; x < ncol; ++x) {
    out.residuals[x] = FBF_TRY(norm(hp + (x * pairs)));
  }
  FBF_TRY(
      gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(ci_block * ncol),
                                            &kOne<double>, relaxed.delta(), 1, out.ci.data(), 1)));
  if (rotates) {
    out.theta = FBF_TRY(read_rotations(res, roots, rotating, ncol, out.ci.data()));
  }
  FBF_TRY(project_off_roots(res, roots, dim, roots_count * ncol, out.ci.data(), overlaps.data()));
  FBF_TRY(res.sync());
  return std::move(out);
}

// The unreduced equations, symmetric with the CI rows scaled by D_r = 2 w_r:
//   [ H        H_oc     ] [z_o]     [ g     ]
//   [ D H_co   D H_cc   ] [z_c] = - [ D P y ],
// H_co x = P H'(x) c_r and H_cc = P (H_0 - E_r) P, by preconditioned
// conjugate gradients over both blocks at once.
Result<ZVectors> zvectors_coupled(DeviceResources &res, RelaxedHessian &relaxed,
                                  const response::Hessian &hessian, const ci::Roots &roots,
                                  const size_t ncol, const double *const gradients,
                                  const double *const ci_rhs, const double tolerance,
                                  const int32_t max_iterations,
                                  const std::span<const double> theta_rhs, ZVectors &out) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t n_pairs = relaxed.size();
  const auto pairs = static_cast<size_t>(n_pairs);
  const size_t ci_block = roots.vectors.num_elements();
  const size_t roots_count = roots.energies.size();
  const size_t dim = ci_block / roots_count;
  const size_t block = pairs * ncol;
  const size_t ci_all = ci_block * ncol;
  const size_t count = roots_count * ncol;
  const auto ncol32 = static_cast<int32_t>(ncol);
  const auto ci_block32 = static_cast<int32_t>(ci_block);
  const auto dim32 = static_cast<int32_t>(dim);

  // The orbital block's R, P, A P (gathered), S and the gathered P, and
  // diag(H)^-1; the CI block's R, P, A P (gathered, then scaled by D), S,
  // gathered P and P y, its preconditioner and D per vector; and the roots'
  // overlaps. Held across the products, which take the workspace.
  Buffers::device<double> orbital((5 * block) + pairs, res.device());
  Buffers::device<double> ci((7 * ci_all) + ci_block + count, res.device());
  Buffers::device<double> overlaps(roots_count * count, res.device());
  double *const z_o = out.orbital.data();
  double *const r_o = orbital.data();
  double *const p_o = r_o + block;
  double *const ap_o = p_o + block;
  double *const s_o = ap_o + block;
  double *const xs_o = s_o + block;
  double *const inverse_o = xs_o + block;
  double *const z_c = out.ci.data();
  double *const r_c = ci.data();
  double *const p_c = r_c + ci_all;
  double *const ap_c = p_c + ci_all;
  double *const dap_c = ap_c + ci_all;
  double *const s_c = dap_c + ci_all;
  double *const xs_c = s_c + ci_all;
  double *const py = xs_c + ci_all;
  double *const inverse_c = py + ci_all;
  double *const scales = inverse_c + ci_block;

  // D per vector (vector x nroots + r at 2 w_r), and the CI preconditioner
  // 1 / (D_r max(|diag(H_0) - E_r|, floor)), built on the host.
  const std::span<const double> weights = relaxed.weights();
  {
    std::vector<double> d(count);
    for (size_t v = 0; v < count; ++v) {
      d[v] = 2.0 * weights[v % roots_count];
    }
    FBF_TRY(relaxed.ci_diagonal(res, inverse_c));
    std::vector<double> h0(dim);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h0.data(), inverse_c, dim * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    std::vector<double> inverse(ci_block);
    for (size_t root = 0; root < roots_count; ++root) {
      for (size_t i = 0; i < dim; ++i) {
        inverse[(root * dim) + i] =
            1.0 /
            (d[root] * std::max(std::abs(h0[i] - roots.energies[root]), kCoupledCiDiagonalFloor));
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(inverse_c, inverse.data(), ci_block * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scales, d.data(), count * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(res.sync());
  }
  FBF_TRY(inverse_orbital_diagonal(res, hessian, inverse_o));

  // The rotations, on the host (a few doubles a column): per pair r < s of
  // unequal weights D_rs = 2 (w_r - w_s) scales its row to the energy's,
  // making the operator symmetric, and 1 / (D_rs (E_s - E_r)) is its exact
  // diagonal's inverse; both zero for the other pairs, which stay at zero.
  const std::vector<bool> rotating = rotating_pairs(weights);
  const bool rotates = std::ranges::contains(rotating, true);
  const size_t n_theta = rotating.size();
  const size_t theta_all = n_theta * ncol;
  std::vector<double> d_theta(n_theta, 0.0);
  std::vector<double> inverse_theta(n_theta, 0.0);
  {
    size_t pair = 0;
    for (size_t r = 0; r < roots_count; ++r) {
      for (size_t s = r + 1; s < roots_count; ++s, ++pair) {
        if (!rotating[pair]) {
          continue;
        }
        d_theta[pair] = 2.0 * (weights[r] - weights[s]);
        const double diagonal = d_theta[pair] * (roots.energies[s] - roots.energies[r]);
        if (!(diagonal > 0.0)) {
          return err_config("mcscf::zvectors (Coupled): roots {} and {} at weights {:g} and {:g}, "
                            "energies {:.10f} and {:.10f}: the rotation between them is not "
                            "positive, (w_r - w_s)(E_s - E_r) = {:.3e} (the weights must fall as "
                            "the energies rise; Schur solves it)",
                            r, s, weights[r], weights[s], roots.energies[r], roots.energies[s],
                            0.5 * diagonal);
        }
        inverse_theta[pair] = 1.0 / diagonal;
      }
    }
  }
  std::vector<double> &z_t = out.theta;
  std::vector<double> r_t(theta_all, 0.0);
  std::vector<double> p_t(theta_all, 0.0);
  std::vector<double> s_t(theta_all, 0.0);
  std::vector<double> xs_t(theta_all, 0.0);
  std::vector<double> ap_t(theta_all, 0.0);
  const auto theta_dot = [&](const std::vector<double> &u, const std::vector<double> &v,
                             const size_t x) {
    double sum = 0.0;
    for (size_t i = x * n_theta; i < (x + 1) * n_theta; ++i) {
      sum += u[i] * v[i];
    }
    return sum;
  };
  // dst <- the first n vectors of src, each scaled by its D_r.
  const auto scale_rows = [&](const double *src, double *dst, size_t n) -> Status {
    return gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim32,
                                                 static_cast<int32_t>(n), src, dim32, scales, 1,
                                                 dst, dim32));
  };
  const auto precondition = [&]() -> Status {
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_pairs, ncol32,
                                                  r_o, n_pairs, inverse_o, 1, s_o, n_pairs)));
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, ci_block32, ncol32,
                                                  r_c, ci_block32, inverse_c, 1, s_c, ci_block32)));
    for (size_t i = 0; i < theta_all; ++i) {
      s_t[i] = inverse_theta[i % n_theta] * r_t[i];
    }
    return project_off_roots(res, roots, dim, count, s_c, overlaps.data());
  };
  // A column's dot over both blocks.
  const auto dot = [&](const double *u_o, const double *v_o, const double *u_c,
                       const double *v_c) -> Result<double> {
    double orbital_part = 0.0;
    double ci_part = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n_pairs, u_o, 1, v_o, 1, &orbital_part)));
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, ci_block32, u_c, 1, v_c, 1, &ci_part)));
    return orbital_part + ci_part;
  };

  // Z = 0, R = -(g, D P y), P = S = M R.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(z_o, 0, block * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(z_c, 0, ci_all * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_pairs,
                                                ncol32, &kNegOne<double>, gradients, n_pairs,
                                                &kZero<double>, gradients, n_pairs, r_o, n_pairs)));
  if (ci_rhs == nullptr) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(py, 0, ci_all * sizeof(double), stream)));
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(r_c, 0, ci_all * sizeof(double), stream)));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(py, ci_rhs, ci_all * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(project_off_roots(res, roots, dim, count, py, overlaps.data()));
    FBF_TRY(scale_rows(py, r_c, count));
    FBF_TRY(gpu_status(
        wwr::scal<double, int32_t>(blas, static_cast<int32_t>(ci_all), &kNegOne<double>, r_c, 1)));
  }
  // R_theta = -D t.
  if (!theta_rhs.empty()) {
    for (size_t i = 0; i < theta_all; ++i) {
      r_t[i] = -d_theta[i % n_theta] * theta_rhs[i];
    }
  }
  FBF_TRY(precondition());
  p_t = s_t;
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p_o, s_o, block * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(p_c, s_c, ci_all * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  std::vector<double> rs(ncol);
  std::vector<double> r_norm(ncol);
  for (size_t x = 0; x < ncol; ++x) {
    const size_t at = x * pairs;
    const size_t at_c = x * ci_block;
    rs[x] = FBF_TRY(dot(r_o + at, s_o + at, r_c + at_c, s_c + at_c)) + theta_dot(r_t, s_t, x);
    r_norm[x] = std::sqrt(FBF_TRY(dot(r_o + at, r_o + at, r_c + at_c, r_c + at_c)) +
                          theta_dot(r_t, r_t, x));
  }

  for (int32_t sweeps = 0;; ++sweeps) {
    const std::vector<size_t> active =
        FBF_TRY(active_columns(r_norm, tolerance, sweeps, max_iterations, "Coupled"));
    if (active.empty()) {
      break;
    }
    const size_t n_active = active.size();
    for (size_t i = 0; i < n_active; ++i) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(xs_o + (i * pairs), p_o + (active[i] * pairs),
                                             pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                             stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(xs_c + (i * ci_block), p_c + (active[i] * ci_block),
                                             ci_block * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      std::copy_n(p_t.begin() + static_cast<std::ptrdiff_t>(active[i] * n_theta), n_theta,
                  xs_t.begin() + static_cast<std::ptrdiff_t>(i * n_theta));
    }
    const size_t active_theta = rotates ? n_active * n_theta : 0;
    FBF_TRY(relaxed.apply_coupled(res, static_cast<int32_t>(n_active), xs_o, xs_c, ap_o, ap_c,
                                  std::span<const double>(xs_t).first(active_theta),
                                  std::span<double>(ap_t).first(active_theta)));
    FBF_TRY(scale_rows(ap_c, dap_c, n_active * roots_count));
    // D A p's rotation rows, in place.
    for (size_t i = 0; i < active_theta; ++i) {
      ap_t[i] *= d_theta[i % n_theta];
    }
    for (size_t i = 0; i < n_active; ++i) {
      const size_t x = active[i];
      const size_t at = x * pairs;
      const size_t at_c = x * ci_block;
      const double *const apx = ap_o + (i * pairs);
      const double *const apx_c = dap_c + (i * ci_block);
      const size_t at_t = x * n_theta;
      const size_t ap_at = i * n_theta;
      ++out.iterations[x];
      double pap_theta = 0.0;
      if (rotates) {
        for (size_t t = 0; t < n_theta; ++t) {
          pap_theta += p_t[at_t + t] * ap_t[ap_at + t];
        }
      }
      const double pap = FBF_TRY(dot(p_o + at, apx, p_c + at_c, apx_c)) + pap_theta;
      if (!(pap > 0.0)) {
        return err_divergence("mcscf::zvectors (Coupled): p.Ap = {:.3e} in column {} after {} "
                              "products: the coupled Hessian is not positive definite along "
                              "it (not a minimum, or a lower root of another spin)",
                              pap, x, out.iterations[x]);
      }
      const double alpha = rs[x] / pap;
      const double neg_alpha = -alpha;
      FBF_TRY(
          gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &alpha, p_o + at, 1, z_o + at, 1)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, ci_block32, &alpha, p_c + at_c, 1, z_c + at_c, 1)));
      FBF_TRY(
          gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &neg_alpha, apx, 1, r_o + at, 1)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, ci_block32, &neg_alpha, apx_c, 1, r_c + at_c, 1)));
      if (rotates) {
        for (size_t t = 0; t < n_theta; ++t) {
          z_t[at_t + t] += alpha * p_t[at_t + t];
          r_t[at_t + t] -= alpha * ap_t[ap_at + t];
        }
      }
      r_norm[x] = std::sqrt(FBF_TRY(dot(r_o + at, r_o + at, r_c + at_c, r_c + at_c)) +
                            theta_dot(r_t, r_t, x));
    }
    FBF_TRY(precondition());
    for (const size_t x : active) {
      const size_t at = x * pairs;
      const size_t at_c = x * ci_block;
      const double rs_new =
          FBF_TRY(dot(r_o + at, s_o + at, r_c + at_c, s_c + at_c)) + theta_dot(r_t, s_t, x);
      const double beta = rs_new / rs[x];
      for (size_t t = x * n_theta; t < (x + 1) * n_theta; ++t) {
        p_t[t] = s_t[t] + (beta * p_t[t]);
      }
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n_pairs, &beta, p_o + at, 1)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>, s_o + at, 1, p_o + at, 1)));
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, ci_block32, &beta, p_c + at_c, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, ci_block32, &kOne<double>, s_c + at_c, 1,
                                                    p_c + at_c, 1)));
      rs[x] = rs_new;
    }
  }

  // The true residuals of the unscaled equations, from one more product:
  // (H z_o + H_oc z_c + g, P ((H_0 - E) z_c + H' c + y), and the rotations'
  // rows plus t).
  FBF_TRY(project_off_roots(res, roots, dim, count, z_c, overlaps.data()));
  const size_t all_theta = rotates ? theta_all : 0;
  FBF_TRY(relaxed.apply_coupled(res, ncol32, z_o, z_c, ap_o, ap_c,
                                std::span<const double>(z_t).first(all_theta),
                                std::span<double>(ap_t).first(all_theta)));
  for (size_t i = 0; i < all_theta; ++i) {
    ap_t[i] =
        d_theta[i % n_theta] == 0.0 ? 0.0 : ap_t[i] + (theta_rhs.empty() ? 0.0 : theta_rhs[i]);
  }
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(block), &kOne<double>,
                                                gradients, 1, ap_o, 1)));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(ci_all), &kOne<double>,
                                                py, 1, ap_c, 1)));
  for (size_t x = 0; x < ncol; ++x) {
    out.residuals[x] = std::sqrt(FBF_TRY(dot(ap_o + (x * pairs), ap_o + (x * pairs),
                                             ap_c + (x * ci_block), ap_c + (x * ci_block))) +
                                 (rotates ? theta_dot(ap_t, ap_t, x) : 0.0));
  }
  FBF_TRY(res.sync());
  return std::move(out);
}

Result<ZVectors> zvectors(DeviceResources &res, RelaxedHessian &relaxed,
                          const response::Hessian &hessian, const ci::Roots &roots,
                          const int32_t ncol, const double *const gradients,
                          const double *const ci_rhs, const CpSolver solver, const double tolerance,
                          const int32_t max_iterations, const std::span<const double> theta_rhs) {
  const int32_t n_pairs = relaxed.size();
  const size_t roots_count = roots.energies.size();
  const size_t n_theta = roots_count * (roots_count - 1) / 2;
  if (hessian.size() != n_pairs || roots_count == 0 ||
      roots.vectors.num_elements() % roots_count != 0 || !(tolerance > 0.0) || max_iterations < 1 ||
      ncol < 1 || (!theta_rhs.empty() && theta_rhs.size() != n_theta * static_cast<size_t>(ncol))) {
    return err_config("mcscf::zvectors: needs H of H~'s {} pairs, at least one root as a whole "
                      "number of vectors, tolerance > 0, max_iterations >= 1, ncol >= 1 and no "
                      "rotation right-hand side or one per pair of roots a column (H {}, {} roots "
                      "in {} doubles, tolerance {:g}, max_iterations {}, ncol {}, {} rotations)",
                      n_pairs, hessian.size(), roots_count, roots.vectors.num_elements(), tolerance,
                      max_iterations, ncol, theta_rhs.size());
  }
  if (solver == CpSolver::Coupled &&
      std::ranges::any_of(relaxed.weights(), [](double w) { return !(w > 0.0); })) {
    return err_config("mcscf::zvectors (Coupled): every weight must be positive, its CI rows "
                      "being scaled by 2 w_r to make the operator symmetric");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto columns = static_cast<size_t>(ncol);
  ZVectors out{Buffers::device<double>(std::max<size_t>(static_cast<size_t>(n_pairs), 1) * columns,
                                       res.device()),
               Buffers::device<double>(roots.vectors.num_elements() * columns, res.device()),
               std::vector<double>(n_theta * columns, 0.0), std::vector<int32_t>(columns, 0),
               std::vector<double>(columns, 0.0)};
  switch (solver) {
  case CpSolver::PerColumn:
    return zvectors_per_column(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                               max_iterations, theta_rhs, out);
  case CpSolver::Schur:
    return zvectors_schur(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                          max_iterations, theta_rhs, out);
  case CpSolver::Coupled:
    return zvectors_coupled(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                            max_iterations, theta_rhs, out);
  }
  return err_config("mcscf::zvectors: unknown solver {}", static_cast<int32_t>(solver));
}

} // namespace free_breakfast::mcscf
