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

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// A pair whose diag(H) is below this is left unscaled by the preconditioner.
// At a minimum diag(H) >= diag(H~) > 0, so it guards only roundoff.
constexpr double kZvectorDiagonalFloor = 1e-8;

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

  // diag(H)^-1, floored, built on the host once.
  std::vector<double> scale(pairs);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(scale.data(), hessian.diagonal().data(),
                                     pairs * sizeof(double), wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  for (double &d : scale) {
    d = d > kZvectorDiagonalFloor ? 1.0 / d : 1.0;
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(inverse_diagonal, scale.data(), pairs * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
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

} // namespace free_breakfast::mcscf
