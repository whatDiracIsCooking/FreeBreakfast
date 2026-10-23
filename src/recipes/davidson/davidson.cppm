/// @file
/// @brief Block Davidson for the lowest eigenpairs of a symmetric operator
/// known only as a LinearOperator, on the GPU over wwr's BLAS.
///
/// The README has the iteration, the buffers and why none of them is the
/// workspace's.

export module free_breakfast.recipes.davidson;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // gemm, axpy, dot, nrm2, scal, ScopedPointerMode, CreatePolicy
import free_breakfast.core.resources;
import free_breakfast.recipes.diagonalize;
export import free_breakfast.recipes.linear_operator;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The knobs of one davidson_solve call.
struct DavidsonOptions {
  /// @brief A root converges when ||A x - theta x||_2 <= residual_tolerance *
  /// max(|theta|, ||H||_2), ||H||_2 being max |theta| over the subspace
  /// spectrum: relative, so it means the same at any scale of A.
  double residual_tolerance = 1e-8;
  /// @brief Subspace expansions before giving up. At least 1.
  int32_t max_iterations = 100;
  /// @brief A re-orthogonalized correction at or below this norm has nothing
  /// left outside the subspace and is dropped. Kept well below
  /// residual_tolerance: a nearly converged root's correction shrinks with its
  /// residual, and a floor near the tolerance would drop real new directions.
  double linear_dependence_floor = 1e-12;
};

/// @brief What a converged davidson_solve did.
struct DavidsonInfo {
  /// @brief Rayleigh-Ritz steps run, the converging one included.
  int32_t iterations = 0;
  /// @brief The largest root residual norm at the last Rayleigh-Ritz step.
  double max_residual_norm = 0.0;
};

/// @brief `precondition(res, theta, residual, correction)`: turn the n x k
/// residual block into a correction block of the same shape (both device,
/// column-major, leading dimension n), given the k current Ritz values
/// @p theta on the host. The diagonal correction r / (theta - diag(A)) is the
/// usual choice. Like LinearOperator::apply, it runs on res.stream() and may
/// take res.device_workspace().
template <class F, class T>
concept DavidsonPreconditioner = requires(
    const F &f, DeviceResources &res, std::span<const T> theta, const T *residual, T *correction) {
  { f(res, theta, residual, correction) } -> std::convertible_to<Status>;
};

/// @brief Converge the @p n_roots lowest eigenpairs of the symmetric @p op
/// from the @p guess_count columns of @p guess.
///
/// Each iteration applies @p op once, to only the columns new to the
/// subspace, diagonalizes the projected matrix H = V^T A V, and expands V by
/// the preconditioned residuals of the roots not yet converged,
/// re-orthogonalized against V and each other. V collapses onto the current
/// Ritz vectors when a full new block would not fit in @p max_subspace
/// columns.
///
/// Everything is queued on res.stream() with res.blas() held in host pointer
/// mode, including the @p op and @p precondition calls, and each iteration
/// synchronizes the stream to read the Ritz values and residual norms. The
/// subspace lives in pool buffers of the call's own, allocated on entry: @p op
/// and @p precondition may take res.device_workspace(), and diagonalize takes
/// it for syevd. The device res.dev_idx() names is made current for the call.
///
/// @param op            Applied as op.apply(res, k, x, y). Must be symmetric.
/// @param precondition  Turns residuals into corrections.
/// @param n             The order of @p op.
/// @param n_roots       Roots wanted, at least 1.
/// @param max_subspace  Columns of V, at least 2 * @p n_roots: a collapsed
///                      subspace always has room for a full new block.
/// @param guess         Device, n x guess_count, leading dimension n, with
///                      orthonormal columns.
/// @param guess_count   n_roots <= guess_count <= max_subspace.
/// @param eigenvalues   Out: device, the n_roots lowest Ritz values, ascending.
/// @param eigenvectors  Out: device, n x n_roots, leading dimension n, their
///                      Ritz vectors.
/// @return The iteration count and final residual on convergence. Otherwise
///         InvalidConfig for a shape outside the bounds above;
///         MaxIterationsExceeded once max_iterations run out; Divergence when
///         no correction survives re-orthogonalization, so the subspace can no
///         longer grow; or the first failing call's error, @p op's and
///         @p precondition's included. After MaxIterationsExceeded or
///         Divergence, @p eigenvalues and @p eigenvectors hold the last
///         iteration's Ritz pairs.
template <real_fp T, LinearOperator<T> Op, DavidsonPreconditioner<T> Precondition>
[[nodiscard]] Result<DavidsonInfo>
davidson_solve(DeviceResources &res, Op &op, const Precondition &precondition, const int32_t n,
               const int32_t n_roots, const int32_t max_subspace, const T *guess,
               const int32_t guess_count, T *eigenvalues, T *eigenvectors,
               const DavidsonOptions &options = {}) {
  if (n < 1 || n_roots < 1 || max_subspace < 2 * n_roots || guess_count < n_roots ||
      guess_count > max_subspace || options.max_iterations < 1) {
    return err_config("davidson_solve: needs n >= 1, n_roots >= 1, max_subspace >= 2 * n_roots, "
                      "n_roots <= guess_count <= max_subspace and max_iterations >= 1 (n = {}, "
                      "n_roots = {}, max_subspace = {}, guess_count = {}, max_iterations = {})",
                      n, n_roots, max_subspace, guess_count, options.max_iterations);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nz = static_cast<size_t>(n);
  const auto roots = static_cast<size_t>(n_roots);
  const auto columns = static_cast<size_t>(max_subspace);
  const size_t block_bytes = nz * roots * sizeof(T);
  const T tolerance = static_cast<T>(options.residual_tolerance);
  const T floor = static_cast<T>(options.linear_dependence_floor);

  // The level-1 reductions write host results and every scalar is a host
  // value, so host pointer mode for the call; the previous mode is restored.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};

  // Scratch held across op.apply, precondition and diagonalize, each of which
  // may take res.device_workspace(), so the call's own pool buffers. Their
  // frees are ordered on the stream after the work that reads them.
  Buffers::device<T> v(nz * columns, res.device());  // the orthonormal basis V
  Buffers::device<T> av(nz * columns, res.device()); // A V, in lockstep with V
  // H = V^T A V, packed at the current dimension (diagonalize's layout), then
  // its eigenvectors S; and its ascending eigenvalues.
  Buffers::device<T> h(columns * columns, res.device());
  Buffers::device<T> ritz(columns, res.device());
  Buffers::device<T> ritz_av(nz * roots, res.device()); // A X for the Ritz vectors X
  Buffers::device<T> residual(nz * roots, res.device());
  Buffers::device<T> correction(nz * roots, res.device());
  Buffers::device<T> projection(columns * roots, res.device()); // V^T C

  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), guess,
                                         nz * static_cast<size_t>(guess_count) * sizeof(T),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));

  // The last Rayleigh-Ritz step's values, copied out on every return after one.
  const auto write_eigenvalues = [&]() -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(eigenvalues, ritz.data(), roots * sizeof(T),
                                          wwr::wwrMemcpyDeviceToDevice, stream));
  };

  std::vector<T> theta(roots);
  std::vector<bool> converged(roots);
  DavidsonInfo info;
  int32_t dim = guess_count;
  int32_t filled = 0; // columns of av already holding A v
  for (int32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    info.iterations = iteration;

    // 1. Extend A V over the new columns only: one apply per iteration.
    if (dim > filled) {
      const size_t offset = static_cast<size_t>(filled) * nz;
      FBF_TRY(op.apply(res, dim - filled, v.data() + offset, av.data() + offset));
      filled = dim;
    }

    // 2. Rayleigh-Ritz: H = V^T A V, then H = S diag(ritz) S^T, which syncs.
    FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, dim,
                                             n, &kOne<T>, v.data(), n, av.data(), n, &kZero<T>,
                                             h.data(), dim)));
    FBF_TRY(diagonalize<T>(res, dim, h.data(), ritz.data()));

    // 3. Ritz vectors X = V S and their images A X = (A V) S, from the leading
    //    n_roots columns of S: A is never applied to X.
    FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots,
                                             dim, &kOne<T>, v.data(), n, h.data(), dim, &kZero<T>,
                                             eigenvectors, n)));
    FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots,
                                             dim, &kOne<T>, av.data(), n, h.data(), dim, &kZero<T>,
                                             ritz_av.data(), n)));
    // ||H||_2 is max |theta| over the ascending spectrum: one of its two ends.
    T theta_top{};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta.data(), ritz.data(), roots * sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&theta_top, ritz.data() + (dim - 1), sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    const T h_norm = std::max(std::abs(theta.front()), std::abs(theta_top));

    // 4. Residuals R = A X - X diag(theta), and the relative test per root.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(residual.data(), ritz_av.data(), block_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    bool all_converged = true;
    T max_norm{};
    for (size_t i = 0; i < roots; ++i) {
      const T neg_theta = -theta[i];
      FBF_TRY(gpu_status(wwr::axpy<T, int32_t>(blas, n, &neg_theta, eigenvectors + (i * nz), 1,
                                               residual.data() + (i * nz), 1)));
      T norm{};
      FBF_TRY(gpu_status(wwr::nrm2<T, int32_t>(blas, n, residual.data() + (i * nz), 1, &norm)));
      converged[i] = norm <= tolerance * std::max(std::abs(theta[i]), h_norm);
      all_converged = all_converged && converged[i];
      max_norm = std::max(max_norm, norm);
    }
    info.max_residual_norm = static_cast<double>(max_norm);
    FBF_LOG_DEBUG("davidson_solve: iteration {}, subspace {}, max residual {:.3e}", iteration, dim,
                  info.max_residual_norm);
    if (all_converged) {
      FBF_TRY(write_eigenvalues());
      return info;
    }

    // 5. Corrections for the whole block; a converged root's is dropped below.
    FBF_TRY(precondition(res, std::span<const T>{theta}, residual.data(), correction.data()));

    // 6. Collapse onto the Ritz pairs if a full new block would not fit.
    //    max_subspace >= 2 * n_roots leaves room for one after.
    if (dim + n_roots > max_subspace) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), eigenvectors, block_bytes,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(av.data(), ritz_av.data(), block_bytes,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      dim = n_roots;
      filled = n_roots;
    }

    // 7. C -= V (V^T C), twice: one pass loses orthogonality once V is large.
    for (int32_t pass = 0; pass < 2; ++pass) {
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim,
                                               n_roots, n, &kOne<T>, v.data(), n, correction.data(),
                                               n, &kZero<T>, projection.data(), dim)));
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots, dim, &kNegOne<T>, v.data(), n,
          projection.data(), dim, &kOne<T>, correction.data(), n)));
    }

    // 8. Modified Gram-Schmidt of the unconverged corrections against each
    //    other, each survivor normalized and appended to V.
    int32_t kept = 0;
    for (size_t i = 0; i < roots; ++i) {
      if (converged[i]) {
        continue;
      }
      T *candidate = correction.data() + (i * nz);
      for (int32_t j = 0; j < kept; ++j) {
        const T *accepted = v.data() + (static_cast<size_t>(dim + j) * nz);
        T overlap{};
        FBF_TRY(gpu_status(wwr::dot<T, int32_t>(blas, n, accepted, 1, candidate, 1, &overlap)));
        const T neg_overlap = -overlap;
        FBF_TRY(
            gpu_status(wwr::axpy<T, int32_t>(blas, n, &neg_overlap, accepted, 1, candidate, 1)));
      }
      T norm{};
      FBF_TRY(gpu_status(wwr::nrm2<T, int32_t>(blas, n, candidate, 1, &norm)));
      if (norm <= floor) {
        continue;
      }
      const T inv_norm = T{1} / norm;
      FBF_TRY(gpu_status(wwr::scal<T, int32_t>(blas, n, &inv_norm, candidate, 1)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data() + (static_cast<size_t>(dim + kept) * nz),
                                             candidate, nz * sizeof(T),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      ++kept;
    }
    if (kept == 0) {
      FBF_TRY(write_eigenvalues());
      return err_divergence("davidson_solve: no correction survived re-orthogonalization at "
                            "iteration {}, so the subspace cannot grow (max residual {:.3e})",
                            iteration, info.max_residual_norm);
    }
    dim += kept;
  }
  FBF_TRY(write_eigenvalues());
  return err_max_iterations_exceeded(
      "davidson_solve: {} roots not converged in {} iterations (max residual {:.3e}, tolerance "
      "{:.3e})",
      n_roots, options.max_iterations, info.max_residual_norm, options.residual_tolerance);
}

} // namespace free_breakfast
