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

Result<ZVector> zvector(DeviceResources &res, RelaxedHessian &relaxed,
                        const response::Hessian &hessian, const ci::Roots &roots,
                        const double *const gradient, const double *const ci_rhs,
                        const double tolerance, const int32_t max_iterations) {
  const int32_t n_pairs = relaxed.size();
  const size_t roots_count = roots.energies.size();
  if (hessian.size() != n_pairs || roots_count == 0 ||
      roots.vectors.num_elements() % roots_count != 0 || !(tolerance > 0.0) || max_iterations < 1) {
    return err_config("mcscf::zvector: needs H of H~'s {} pairs, at least one root as a whole "
                      "number of vectors, tolerance > 0 and max_iterations >= 1 (H {}, {} roots "
                      "in {} doubles, tolerance {:g}, max_iterations {})",
                      n_pairs, hessian.size(), roots_count, roots.vectors.num_elements(), tolerance,
                      max_iterations);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto pairs = static_cast<size_t>(n_pairs);
  const size_t dim = roots.vectors.num_elements() / roots_count;
  const auto dim32 = static_cast<int32_t>(dim);
  const auto nroots = static_cast<int32_t>(roots_count);

  ZVector out{Buffers::device<double>(std::max<size_t>(pairs, 1), res.device()),
              Buffers::device<double>(dim * roots_count, res.device()), 0, 0.0};
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
                                     const int32_t max_iterations, ZVectors &out) {
  const auto pairs = static_cast<size_t>(relaxed.size());
  const size_t ci_block = roots.vectors.num_elements();
  for (size_t x = 0; x < ncol; ++x) {
    const ZVector z = FBF_TRY(zvector(res, relaxed, hessian, roots, gradients + (x * pairs),
                                      ci_rhs == nullptr ? nullptr : ci_rhs + (x * ci_block),
                                      tolerance, max_iterations));
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
                                const int32_t max_iterations, ZVectors &out) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t n_pairs = relaxed.size();
  const auto pairs = static_cast<size_t>(n_pairs);
  const size_t ci_block = roots.vectors.num_elements();
  const size_t roots_count = roots.energies.size();
  const size_t dim = ci_block / roots_count;
  const size_t block = pairs * ncol;
  const auto ncol32 = static_cast<int32_t>(ncol);

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
                                  const int32_t max_iterations, ZVectors &out) {
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
  FBF_TRY(precondition());
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p_o, s_o, block * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(p_c, s_c, ci_all * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  std::vector<double> rs(ncol);
  std::vector<double> r_norm(ncol);
  for (size_t x = 0; x < ncol; ++x) {
    const size_t at = x * pairs;
    const size_t at_c = x * ci_block;
    rs[x] = FBF_TRY(dot(r_o + at, s_o + at, r_c + at_c, s_c + at_c));
    r_norm[x] = std::sqrt(FBF_TRY(dot(r_o + at, r_o + at, r_c + at_c, r_c + at_c)));
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
    }
    FBF_TRY(relaxed.apply_coupled(res, static_cast<int32_t>(n_active), xs_o, xs_c, ap_o, ap_c));
    FBF_TRY(scale_rows(ap_c, dap_c, n_active * roots_count));
    for (size_t i = 0; i < n_active; ++i) {
      const size_t x = active[i];
      const size_t at = x * pairs;
      const size_t at_c = x * ci_block;
      const double *const apx = ap_o + (i * pairs);
      const double *const apx_c = dap_c + (i * ci_block);
      ++out.iterations[x];
      const double pap = FBF_TRY(dot(p_o + at, apx, p_c + at_c, apx_c));
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
      r_norm[x] = std::sqrt(FBF_TRY(dot(r_o + at, r_o + at, r_c + at_c, r_c + at_c)));
    }
    FBF_TRY(precondition());
    for (const size_t x : active) {
      const size_t at = x * pairs;
      const size_t at_c = x * ci_block;
      const double rs_new = FBF_TRY(dot(r_o + at, s_o + at, r_c + at_c, s_c + at_c));
      const double beta = rs_new / rs[x];
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
  // (H z_o + H_oc z_c + g, P ((H_0 - E) z_c + H' c + y)).
  FBF_TRY(project_off_roots(res, roots, dim, count, z_c, overlaps.data()));
  FBF_TRY(relaxed.apply_coupled(res, ncol32, z_o, z_c, ap_o, ap_c));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(block), &kOne<double>,
                                                gradients, 1, ap_o, 1)));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(ci_all), &kOne<double>,
                                                py, 1, ap_c, 1)));
  for (size_t x = 0; x < ncol; ++x) {
    out.residuals[x] = std::sqrt(FBF_TRY(
        dot(ap_o + (x * pairs), ap_o + (x * pairs), ap_c + (x * ci_block), ap_c + (x * ci_block))));
  }
  FBF_TRY(res.sync());
  return std::move(out);
}

Result<ZVectors> zvectors(DeviceResources &res, RelaxedHessian &relaxed,
                          const response::Hessian &hessian, const ci::Roots &roots,
                          const int32_t ncol, const double *const gradients,
                          const double *const ci_rhs, const CpSolver solver, const double tolerance,
                          const int32_t max_iterations) {
  const int32_t n_pairs = relaxed.size();
  const size_t roots_count = roots.energies.size();
  if (hessian.size() != n_pairs || roots_count == 0 ||
      roots.vectors.num_elements() % roots_count != 0 || !(tolerance > 0.0) || max_iterations < 1 ||
      ncol < 1) {
    return err_config("mcscf::zvectors: needs H of H~'s {} pairs, at least one root as a whole "
                      "number of vectors, tolerance > 0, max_iterations >= 1 and ncol >= 1 (H {}, "
                      "{} roots in {} doubles, tolerance {:g}, max_iterations {}, ncol {})",
                      n_pairs, hessian.size(), roots_count, roots.vectors.num_elements(), tolerance,
                      max_iterations, ncol);
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
               std::vector<int32_t>(columns, 0), std::vector<double>(columns, 0.0)};
  switch (solver) {
  case CpSolver::PerColumn:
    return zvectors_per_column(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                               max_iterations, out);
  case CpSolver::Schur:
    return zvectors_schur(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                          max_iterations, out);
  case CpSolver::Coupled:
    return zvectors_coupled(res, relaxed, hessian, roots, columns, gradients, ci_rhs, tolerance,
                            max_iterations, out);
  }
  return err_config("mcscf::zvectors: unknown solver {}", static_cast<int32_t>(solver));
}

} // namespace free_breakfast::mcscf
