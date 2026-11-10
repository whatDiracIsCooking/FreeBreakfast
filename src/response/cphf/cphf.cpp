/// @file
/// @brief cphf: M's product over the density-fitted J and K, one column at a
/// time or (batched) a block of columns at once, as a LinearOperator, and
/// (eps_a - eps_i)^-1 as a second, handed to recipes::pcg_solve.

module free_breakfast.response.cphf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, dgmm, axpy
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.footprint;  // MemoryFootprint, kPoolHeadroomBytes
import free_breakfast.recipes.blas;    // gemm
import free_breakfast.recipes.pcg;
import free_breakfast.integrals.eri.density_fitting.jk;

#include "diagnostics.h"

namespace free_breakfast::response {

// A pair whose eps_a - eps_i is below this is left unscaled by the
// preconditioner. Above an aufbau RHF's gap it guards only a non-aufbau one.
constexpr double kCphfDenominatorFloor = 1e-8;

// One column's product, in doubles: its [T C_o T] (T = C_v U), D_U, J and K.
size_t cphf_single(const size_t nao, const size_t occ) {
  return (3 * nao * occ) + (3 * nao * nao);
}

// A batched product's, per column, in doubles: L = [T C_o] and R = [C_o T]
// (so D_U = 2 L R^T), D_U, J, K and G C_o.
size_t cphf_per_column(const size_t nao, const size_t occ) {
  return (4 * nao * occ) + (3 * nao * nao) + (nao * occ);
}

// The device workspace one column's unbatched J and K take: df_coulomb's
// gamma, or the two-factor df_exchange's X and Y at rank 2 nocc, each a
// region.
size_t cphf_workspace(const int32_t nao, const int32_t naux, const int32_t nocc) {
  if (nao < 1 || naux < 1) {
    return 0;
  }
  const size_t xy = math::align_up(sizeof(double) * static_cast<size_t>(naux) * 2 *
                                       static_cast<size_t>(nocc) * static_cast<size_t>(nao),
                                   DeviceResources::kWorkspaceAlignment);
  return std::max(df_coulomb_footprint(nao, naux).device.workspace, 2 * xy);
}

// The scratch a batched product's J and K take, @p width columns at a time,
// in bytes: the block df_coulomb's Gamma or the block df_exchange's X and Y
// at rank 2 nocc, allocated and freed in turn.
size_t cphf_batch_jk(const int32_t nao, const int32_t naux, const int32_t nocc,
                     const size_t width) {
  const auto w = static_cast<int32_t>(width);
  return std::max(df_coulomb_footprint(nao, naux, w).device.scratch,
                  df_exchange_footprint(nao, naux, 2 * nocc, w).device.scratch);
}

// Everything cphf allocates from the solution on, in bytes, with a batched
// product of @p width columns (0: unbatched): the solution; the held buffer,
// eps_a - eps_i, the preconditioner and the product's own; pcg_solve's
// vectors (recipes/pcg/pcg.cppm), 4 n ncol + 2 ncol doubles; and, batched,
// the J and K scratch. The unbatched J and K take the workspace instead.
size_t cphf_bytes(const int32_t nao, const int32_t naux, const int32_t nocc, const size_t pairs,
                  const size_t ncol, const size_t width) {
  const auto n_ao = static_cast<size_t>(nao);
  const auto occ = static_cast<size_t>(nocc);
  const size_t block = pairs * ncol;
  const size_t own = width > 0 ? width * cphf_per_column(n_ao, occ) : cphf_single(n_ao, occ);
  return (sizeof(double) * (block + (2 * pairs) + own + (4 * block) + (2 * ncol))) +
         (width > 0 ? cphf_batch_jk(nao, naux, nocc, width) : 0);
}

// How many columns one batched build takes: the most, up to @p ncol, whose
// cphf_bytes (every allocation the call has still to make, read before the
// first) fits the memory budget less the pool's fragmentation headroom, and
// that the int32_t BLAS dispatch allows. At least one: cphf_footprint's.
Result<size_t> cphf_width(DeviceResources &res, const int32_t nao, const int32_t naux,
                          const int32_t nocc, const size_t pairs, const size_t ncol) {
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t usable = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  const size_t int32_max = std::numeric_limits<int32_t>::max();
  const auto n_ao = static_cast<size_t>(nao);
  const size_t widest =
      std::min({ncol, int32_max / (n_ao * n_ao),
                naux > 0 ? int32_max / (static_cast<size_t>(naux) * 2 * nocc) : ncol});
  const auto fits = [&](const size_t width) {
    return cphf_bytes(nao, naux, nocc, pairs, ncol, width) <= usable;
  };
  if (widest < 2 || !fits(2)) {
    return size_t{1};
  }
  // The widest in [2, widest] that fits: the bytes grow with the width.
  size_t lo = 2;
  size_t hi = widest;
  while (lo < hi) {
    const size_t mid = lo + ((hi - lo + 1) / 2);
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

MemoryFootprint cphf_footprint(const int32_t nao, const int32_t naux, const int32_t nmo,
                               const int32_t nocc, const int32_t ncol, const bool batched) {
  MemoryFootprint out;
  if (nocc < 1 || nmo <= nocc || ncol < 1 || nao < 1 || naux < 0) {
    return out;
  }
  const size_t pairs = static_cast<size_t>(nmo - nocc) * static_cast<size_t>(nocc);
  const size_t block = pairs * static_cast<size_t>(ncol);
  // The solution, returned; the rest, held together at the peak, scratch,
  // with a batched product at its least, one column.
  out.device.persistent = sizeof(double) * block;
  out.device.scratch =
      cphf_bytes(nao, naux, nocc, pairs, static_cast<size_t>(ncol), batched ? 1 : 0) -
      out.device.persistent;
  out.device.workspace = batched ? 0 : cphf_workspace(nao, naux, nocc);
  return out;
}

Result<Cphf> cphf(DeviceResources &res, const Buffers::device<double> &b, const int32_t naux,
                  const Buffers::device<double> &c, const std::span<const double> energies,
                  const int32_t nocc, const Buffers::device<double> &rhs, const double tolerance,
                  const int32_t max_iterations, const bool batched) {
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
  const size_t nn = nao * nao;
  const size_t occ_size = nao * static_cast<size_t>(nocc);
  // One column's product (cphf_single), or a batched one's per column
  // (cphf_per_column). pcg_solve gathers the columns it applies M to into a
  // block of its own.
  const size_t factors_size = 3 * occ_size;
  const size_t single = cphf_single(nao, static_cast<size_t>(nocc));
  const size_t factor_size = 2 * occ_size;
  const size_t per_column = cphf_per_column(nao, static_cast<size_t>(nocc));

  // How many columns one batched build takes: cphf_width's, read before the
  // first allocation.
  const size_t width = batched ? FBF_TRY(cphf_width(res, n_ao, naux, nocc, pairs, ncol)) : 0;

  Buffers::device<double> solution(block, res.device());
  // eps_a - eps_i and the preconditioner, and one column's product or a
  // batch of width columns'. Held across df_coulomb and df_exchange, which
  // take the workspace unbatched, so its own (docs/style.md "Scratch memory").
  Buffers::device<double> held((2 * pairs) + (batched ? width * per_column : single), res.device());
  double *const delta = held.data();
  double *const inverse_delta = delta + pairs;
  // One column's product.
  double *const factors = inverse_delta + pairs;
  double *const density = factors + factors_size;
  double *const j = density + nn;
  double *const k = j + nn;
  // A batch's, over the same memory.
  double *const left = inverse_delta + pairs;
  double *const right = left + (width * factor_size);
  double *const densities = right + (width * factor_size);
  double *const js = densities + (width * nn);
  double *const ks = js + (width * nn);
  double *const ws = ks + (width * nn);
  const double *const c_occ = c.data();
  const double *const c_vir = c.data() + occ_size;

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
  if (batched) {
    // C_o in every batch column's L and R once: L_x = [T_x C_o] and
    // R_x = [C_o T_x], the T blocks written by each product.
    for (size_t x = 0; x < width; ++x) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(left + (x * factor_size) + occ_size, c_occ,
                                             occ_size * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(right + (x * factor_size), c_occ,
                                             occ_size * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    }
  } else {
    // C_o, the middle of [T C_o T], once: D_U = 2 L R^T with L = [T C_o] and
    // R = [C_o T], the factors' first and last two blocks.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(factors + occ_size, c_occ, occ_size * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }

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
  // Y <- M X for count <= width packed columns, one batched J and K build.
  const auto apply_batch = [&](const double *x, double *y, const size_t count) -> Status {
    const auto cols = static_cast<int32_t>(count);
    const auto factor_stride = static_cast<int64_t>(factor_size);
    const auto pairs_stride = static_cast<int64_t>(pairs);
    const auto nn_stride = static_cast<int64_t>(nn);
    const auto occ_stride = static_cast<int64_t>(occ_size);
    // T_x = C_v U_x into L_x's first block and R_x's second.
    for (double *const t : {left, right + occ_size}) {
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_ao, nocc, n_vir, &kOne<double>, c_vir, n_ao,
          0, x, n_vir, pairs_stride, &kZero<double>, t, n_ao, factor_stride, cols)));
    }
    // D_U = 2 L R^T = 2 (T C_o^T + C_o T^T).
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_ao, n_ao, 2 * nocc, &kTwo<double>, left, n_ao,
        factor_stride, right, n_ao, factor_stride, &kZero<double>, densities, n_ao, nn_stride,
        cols)));
    // G = J[D_U] - K[D_U] / 2 = J[D_U] - K[L R^T].
    FBF_TRY(df_coulomb(res, n_ao, naux, cols, b.data(), densities, js));
    FBF_TRY(df_exchange(res, n_ao, naux, 2 * nocc, cols, b.data(), left, right, ks));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(count * nn),
                                                  &kNegOne<double>, ks, 1, js, 1)));
    // Y = (eps_a - eps_i) X + C_v^T (G C_o): G_x C_o side by side is the
    // nao x (count n_occ) W, and C_v^T W is Y as stored.
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_ao, nocc, n_ao, &kOne<double>, js, n_ao,
        nn_stride, c_occ, n_ao, 0, &kZero<double>, ws, n_ao, occ_stride, cols)));
    FBF_TRY(gpu_status(
        wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n, cols, x, n, delta, 1, y, n)));
    return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_vir, cols * nocc, n_ao, &kOne<double>, c_vir,
                                              ws, &kOne<double>, y));
  };
  // Y <- M X for count packed columns: one column's product at a time, or
  // width columns' at a time by the batched build.
  const auto product = [&](const int32_t count, const double *x, double *y) -> Status {
    const auto columns = static_cast<size_t>(count);
    for (size_t first = 0; first < columns; first += batched ? width : 1) {
      const double *const from = x + (first * pairs);
      double *const to = y + (first * pairs);
      if (batched) {
        FBF_TRY(apply_batch(from, to, std::min(width, columns - first)));
      } else {
        FBF_TRY(apply(from, to));
      }
    }
    return {};
  };
  // M and (eps_a - eps_i)^-1 as pcg_solve's operator and preconditioner.
  struct Product {
    decltype(product) &build;
    Status apply(DeviceResources & /*res*/, const int32_t count, const double *x, double *y) {
      return build(count, x, y);
    }
  };
  struct Jacobi {
    int32_t rows;
    const double *inverse;
    Status apply(DeviceResources &on, const int32_t count, const double *x, double *y) {
      return gpu_status(wwr::dgmm<double, int32_t>(on.blas().get(), wwr::WWRBLAS_SIDE_LEFT, rows,
                                                   count, x, rows, inverse, 1, y, rows));
    }
  };
  Product op{product};
  Jacobi jacobi{n, inverse_delta};

  // U from zero, every column in lockstep: pcg_solve's recurrence is the one
  // this solve ran before it, step for step, so the sweep counts are too.
  PcgOptions options;
  options.tolerance = tolerance;
  options.max_iterations = max_iterations;
  auto solved = pcg_solve(res, op, jacobi, n, static_cast<int32_t>(ncol), rhs.data(),
                          solution.data(), options);
  if (!solved) {
    Error error = std::move(solved).error();
    error.message = std::format(
        "response::cphf: {}{}", error.message,
        error.kind == ErrorKind::Divergence
            ? "; M is the RHF orbital Hessian, so the reference is unstable to a real singlet "
              "rotation"
            : "");
    return std::unexpected(std::move(error));
  }
  return Cphf{std::move(solution), std::move(solved->residuals), solved->sweeps};
}

} // namespace free_breakfast::response
