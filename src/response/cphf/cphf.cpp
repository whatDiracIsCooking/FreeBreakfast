/// @file
/// @brief cphf: M's product over the density-fitted J and K, and
/// preconditioned conjugate gradients over it, every column in lockstep.

module free_breakfast.response.cphf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // dgmm, axpy, dot, nrm2, scal, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.integrals.eri.density_fitting.jk;

#include "diagnostics.h"

namespace free_breakfast::response {

// A pair whose eps_a - eps_i is below this is left unscaled by the
// preconditioner. Above an aufbau RHF's gap it guards only a non-aufbau one.
constexpr double kCphfDenominatorFloor = 1e-8;

Result<Cphf> cphf(DeviceResources &res, const Buffers::device<double> &b, const int32_t naux,
                  const Buffers::device<double> &c, const std::span<const double> energies,
                  const int32_t nocc, const Buffers::device<double> &rhs, const double tolerance,
                  const int32_t max_iterations) {
  const size_t nmo = energies.size();
  const size_t nvir = nmo > static_cast<size_t>(std::max(nocc, 0)) ? nmo - nocc : 0;
  const size_t pairs = nvir * static_cast<size_t>(std::max(nocc, 0));
  if (nocc < 1 || nvir == 0 || naux < 0 || c.num_elements() % nmo != 0 ||
      pairs > static_cast<size_t>(std::numeric_limits<int32_t>::max()) || rhs.num_elements() == 0 ||
      rhs.num_elements() % pairs != 0 || !(tolerance > 0.0) || max_iterations < 1) {
    return err_config("response::cphf: needs 1 <= nocc < nmo, C a whole number of nmo-long "
                      "columns, at least one right-hand side of n_vir n_occ, tolerance > 0 and "
                      "max_iterations >= 1 (nocc {}, nmo {}, naux {}, C {} doubles, {} right-hand "
                      "side doubles, tolerance {:g}, max_iterations {})",
                      nocc, nmo, naux, c.num_elements(), rhs.num_elements(), tolerance,
                      max_iterations);
  }
  const size_t nao = c.num_elements() / nmo;
  if (b.num_elements() != static_cast<size_t>(naux) * nao * nao) {
    return err_config("response::cphf: B holds {} doubles, but naux nao^2 is {}", b.num_elements(),
                      static_cast<size_t>(naux) * nao * nao);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t ncol = rhs.num_elements() / pairs;
  const auto n = static_cast<int32_t>(pairs);
  const auto n_ao = static_cast<int32_t>(nao);
  const auto n_vir = static_cast<int32_t>(nvir);
  const size_t block = pairs * ncol;

  Cphf out{Buffers::device<double>(block, res.device()), std::vector<double>(ncol), 0};
  // The residuals R, the directions P, M P and the preconditioned residuals
  // S, every column side by side; eps_a - eps_i and the preconditioner; and
  // a product's [T C_o T] (T = C_v U), D_U, J and K. Held across df_coulomb
  // and df_exchange, which take the workspace, so its own (docs/style.md
  // "Scratch memory").
  const size_t nn = nao * nao;
  const size_t factors_size = nao * 3 * static_cast<size_t>(nocc);
  Buffers::device<double> held((4 * block) + (2 * pairs) + factors_size + (3 * nn), res.device());
  double *const u = out.solution.data();
  double *const r = held.data();
  double *const p = r + block;
  double *const mp = p + block;
  double *const s = mp + block;
  double *const delta = s + block;
  double *const inverse_delta = delta + pairs;
  double *const factors = inverse_delta + pairs;
  double *const density = factors + factors_size;
  double *const j = density + nn;
  double *const k = j + nn;
  const double *const c_occ = c.data();
  const double *const c_vir = c.data() + (nao * static_cast<size_t>(nocc));
  const size_t occ_size = nao * static_cast<size_t>(nocc);

  // eps_a - eps_i at a + n_vir i, and its floored inverse, built on the host.
  std::vector<double> differences(pairs);
  std::vector<double> scale(pairs);
  for (size_t i = 0; i < static_cast<size_t>(nocc); ++i) {
    for (size_t a = 0; a < nvir; ++a) {
      const double d = energies[nocc + a] - energies[i];
      differences[a + (nvir * i)] = d;
      scale[a + (nvir * i)] = d > kCphfDenominatorFloor ? 1.0 / d : 1.0;
    }
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(delta, differences.data(), pairs * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(inverse_delta, scale.data(), pairs * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // C_o, the middle of [T C_o T], once: D_U = 2 L R^T with L = [T C_o] and
  // R = [C_o T], the factors' first and last two blocks.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(factors + occ_size, c_occ, occ_size * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));

  // y <- M x, one column.
  const auto apply = [&](const double *x, double *y) -> Status {
    // T = C_v U, then D_U = 2 (T C_o^T + C_o T^T).
    FBF_TRY(gpu_status(
        gemm(blas, n_ao, nocc, n_vir, &kOne<double>, c_vir, x, &kZero<double>, factors)));
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(factors + (2 * occ_size), factors, occ_size * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n_ao, n_ao, nocc, &kTwo<double>, factors, c_occ, &kZero<double>, density)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n_ao, n_ao, nocc, &kTwo<double>, c_occ, factors, &kOne<double>, density)));
    // G = J[D_U] - K[D_U] / 2 = J[D_U] - K[L R^T].
    FBF_TRY(df_coulomb(res, n_ao, naux, b.data(), density, j));
    FBF_TRY(df_exchange(res, n_ao, naux, 2 * nocc, b.data(), factors, factors + occ_size, k));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(nn), &kNegOne<double>, k, 1, j, 1)));
    // y = (eps_a - eps_i) x + C_v^T (G C_o), G C_o into K's space.
    FBF_TRY(gpu_status(gemm(blas, n_ao, nocc, n_ao, &kOne<double>, j, c_occ, &kZero<double>, k)));
    FBF_TRY(gpu_status(
        wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n, 1, x, n, delta, 1, y, n)));
    return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_vir, nocc, n_ao, &kOne<double>, c_vir, k,
                                              &kOne<double>, y));
  };
  const auto dot = [&](const double *x, const double *y) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, x, 1, y, 1, &value)));
    return value;
  };
  const auto norm = [&](const double *x) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n, x, 1, &value)));
    return value;
  };
  // S <- (eps_a - eps_i)^-1 R, every column: each element reads only its
  // own, and a stopped column's S is never read.
  const auto precondition = [&]() -> Status {
    return gpu_status(wwr::dgmm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, n, static_cast<int32_t>(ncol), r, n, inverse_delta, 1, s, n));
  };

  // U = 0, R = B, P = S = M^-1 R.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(u, 0, block * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r, rhs.data(), block * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(precondition());
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(p, s, block * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  std::vector<double> rs(ncol);
  std::vector<double> r_norm(ncol);
  for (size_t x = 0; x < ncol; ++x) {
    rs[x] = FBF_TRY(dot(r + (x * pairs), s + (x * pairs)));
    r_norm[x] = FBF_TRY(norm(r + (x * pairs)));
  }

  std::vector<size_t> active;
  for (;;) {
    active.clear();
    for (size_t x = 0; x < ncol; ++x) {
      if (r_norm[x] > tolerance) {
        active.push_back(x);
      }
    }
    if (active.empty()) {
      break;
    }
    if (out.iterations == max_iterations) {
      return err_max_iterations_exceeded(
          "response::cphf: {} of {} columns above {:.3e} after {} sweeps, the largest ||r|| "
          "{:.3e}",
          active.size(), ncol, tolerance, max_iterations, std::ranges::max(r_norm));
    }
    ++out.iterations;
    for (const size_t x : active) {
      const size_t at = x * pairs;
      FBF_TRY(apply(p + at, mp + at));
      const double pmp = FBF_TRY(dot(p + at, mp + at));
      if (!(pmp > 0.0)) {
        return err_divergence("response::cphf: p.Mp = {:.3e} in column {} after {} sweeps: M is "
                              "not positive definite, so the RHF reference is unstable",
                              pmp, x, out.iterations);
      }
      const double alpha = rs[x] / pmp;
      const double neg_alpha = -alpha;
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n, &alpha, p + at, 1, u + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n, &neg_alpha, mp + at, 1, r + at, 1)));
      r_norm[x] = FBF_TRY(norm(r + at));
    }
    FBF_TRY(precondition());
    for (const size_t x : active) {
      const size_t at = x * pairs;
      const double rs_new = FBF_TRY(dot(r + at, s + at));
      const double beta = rs_new / rs[x];
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n, &beta, p + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n, &kOne<double>, s + at, 1, p + at, 1)));
      rs[x] = rs_new;
    }
  }

  // The true residuals M U - B, from one more product per column: the
  // recurrence's R drifts from them in rounding.
  for (size_t x = 0; x < ncol; ++x) {
    const size_t at = x * pairs;
    FBF_TRY(apply(u + at, mp + at));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, n, &kNegOne<double>, rhs.data() + at, 1, mp + at, 1)));
    out.residuals[x] = FBF_TRY(norm(mp + at));
  }
  FBF_TRY(res.sync());
  return out;
}

} // namespace free_breakfast::response
