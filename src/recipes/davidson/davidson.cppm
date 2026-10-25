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
import free_breakfast.core.wwr; // BLAS, wwrMemcpy2DAsync, ScopedPointerMode, CreatePolicy
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.orthogonalize; // orthonormalize
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
  /// @brief A re-orthogonalized correction (or, with
  /// canonical_orthonormalization, a mode of the block of them) at or below
  /// this norm has nothing left outside the subspace and is dropped. Kept well
  /// below
  /// residual_tolerance: a nearly converged root's correction shrinks with its
  /// residual, and a floor near the tolerance would drop real new directions.
  double linear_dependence_floor = 1e-12;
  /// @brief Orthonormalize the new corrections among themselves as one block,
  /// by orthonormalize (canonical orthogonalization), instead of by
  /// Gram-Schmidt, a norm sync per correction. Off by default: even per
  /// iteration, but over the README's N2 bench it took different iteration
  /// paths that were slower on balance. ci_solve's guesses follow it too.
  bool canonical_orthonormalization = false;
};

/// @brief What a converged davidson_solve did.
struct DavidsonInfo {
  /// @brief Rayleigh-Ritz steps run, the converging one included.
  int32_t iterations = 0;
  /// @brief The largest root residual norm at the last Rayleigh-Ritz step.
  double max_residual_norm = 0.0;
};

/// @brief `precondition(res, theta, ritz_vectors, residual, correction)`: turn
/// the n x k residual block into a correction block of the same shape, given
/// the k current Ritz values @p theta on the host and their Ritz vectors X
/// (all three blocks device, column-major, leading dimension n). The diagonal
/// correction r / (theta - diag(A)) is the simplest choice; X is there for
/// Olsen's, M^-1 (r - eps x) with eps = x^T M^-1 r / x^T M^-1 x, which is
/// orthogonal to x (see ci_solve). Like LinearOperator::apply, it runs on
/// res.stream(), with res.blas() in host pointer mode, and may take
/// res.device_workspace().
template <class F, class T>
concept DavidsonPreconditioner =
    requires(const F &f, DeviceResources &res, std::span<const T> theta, const T *ritz_vectors,
             const T *residual, T *correction) {
      { f(res, theta, ritz_vectors, residual, correction) } -> std::convertible_to<Status>;
    };

/// @brief Converge the @p n_roots lowest eigenpairs of the symmetric @p op
/// from the @p guess_count columns of @p guess.
///
/// Each iteration applies @p op once, to only the columns new to the
/// subspace, extends the projected matrix H = V^T A V by those columns' rows
/// alone, diagonalizes it, and expands V by
/// the preconditioned residuals of the roots not yet converged,
/// re-orthogonalized against V and each other. When a full new block would
/// not fit in @p max_subspace columns, V collapses onto the current Ritz
/// vectors X and the previous iteration's X_prev (GD+k), or onto X alone when
/// @p max_subspace < 3 * @p n_roots.
///
/// Everything is queued on res.stream() with res.blas() held in host pointer
/// mode, including the @p op and @p precondition calls. Each iteration
/// synchronizes the stream to read the Ritz values, once for all the residual
/// norms, and once per correction kept, for its norm (with
/// canonical_orthonormalization, twice or four times in orthonormalize
/// instead). The subspace lives in pool buffers of the call's own, allocated
/// on entry: @p op and @p precondition may take res.device_workspace(),
/// diagonalize takes it for syevd and orthonormalize for its product. The
/// device res.dev_idx() names is made current for the call.
///
/// @param op            Applied as op.apply(res, k, x, y). Must be symmetric.
/// @param precondition  Turns residuals into corrections.
/// @param n             The order of @p op.
/// @param n_roots       Roots wanted, at least 1.
/// @param max_subspace  Columns of V, at least 2 * @p n_roots: a collapsed
///                      subspace always has room for a full new block. From
///                      3 * @p n_roots, the collapse also keeps X_prev.
/// @param guess         Device, n x guess_count, leading dimension n, with
///                      orthonormal columns.
/// @param guess_count   n_roots <= guess_count <= max_subspace.
/// @param eigenvalues   Out: device, the n_roots lowest Ritz values, ascending.
/// @param eigenvectors  Out: device, n x n_roots, leading dimension n, their
///                      Ritz vectors.
/// @param previous_eigenvectors Out, when not null: device, n x n_roots,
///                      leading dimension n, the previous iteration's Ritz
///                      vectors X_prev, on every return that writes
///                      @p eigenvectors; @p eigenvectors' own after the first
///                      iteration, which has no previous. Null by default.
/// @return The iteration count and final residual on convergence. Otherwise
///         InvalidConfig for a shape outside the bounds above;
///         MaxIterationsExceeded once max_iterations run out; Divergence when
///         no correction survives re-orthogonalization, so the subspace can no
///         longer grow; or the first failing call's error, @p op's and
///         @p precondition's included. After MaxIterationsExceeded or
///         Divergence, @p eigenvalues and @p eigenvectors hold the last
///         iteration's Ritz pairs, and @p previous_eigenvectors the one's
///         before.
template <real_fp T, LinearOperator<T> Op, DavidsonPreconditioner<T> Precondition>
[[nodiscard]] Result<DavidsonInfo>
davidson_solve(DeviceResources &res, Op &op, const Precondition &precondition, const int32_t n,
               const int32_t n_roots, const int32_t max_subspace, const T *guess,
               const int32_t guess_count, T *eigenvalues, T *eigenvectors,
               const DavidsonOptions &options = {}, T *previous_eigenvectors = nullptr) {
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
  // GD+k: a collapse keeps X_prev too when its up to 2 n_roots columns and a
  // full new block fit.
  const bool keep_previous = max_subspace >= 3 * n_roots;
  // S_prev is tracked for the collapse, or for previous_eigenvectors.
  const bool track_previous = keep_previous || previous_eigenvectors != nullptr;

  // The level-1 reductions write host results and every scalar is a host
  // value, so host pointer mode for the call; the previous mode is restored.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};

  // Scratch held across op.apply, precondition and diagonalize, each of which
  // may take res.device_workspace(), so the call's own pool buffers. Their
  // frees are ordered on the stream after the work that reads them.
  Buffers::device<T> v(nz * columns, res.device());  // the orthonormal basis V
  Buffers::device<T> av(nz * columns, res.device()); // A V, in lockstep with V
  // V^T A V's lower triangle, leading dimension max_subspace, grown a row
  // block per apply; its copy packed at the current dimension (diagonalize's
  // layout), then the copy's eigenvectors S; and their ascending eigenvalues.
  Buffers::device<T> vav(columns * columns, res.device());
  Buffers::device<T> h(columns * columns, res.device());
  Buffers::device<T> ritz(columns, res.device());
  Buffers::device<T> ritz_av(nz * roots, res.device()); // A X for the Ritz vectors X
  Buffers::device<T> residual(nz * roots, res.device());
  Buffers::device<T> correction(nz * roots, res.device());
  Buffers::device<T> projection(columns * roots, res.device()); // V^T C
  Buffers::device<T> norms(roots, res.device());    // the residual norms, read back at once
  Buffers::device<T> overlaps(roots, res.device()); // a correction's on those accepted before it

  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), guess,
                                         nz * static_cast<size_t>(guess_count) * sizeof(T),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));

  // The last Rayleigh-Ritz step's values, copied out on every return after one.
  const auto write_eigenvalues = [&]() -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(eigenvalues, ritz.data(), roots * sizeof(T),
                                          wwr::wwrMemcpyDeviceToDevice, stream));
  };

  std::vector<T> theta(roots);
  std::vector<T> residual_norms(roots);
  std::vector<bool> converged(roots);
  // For the GD+k collapse and previous_eigenvectors, on the host: this
  // iteration's S (its leading n_roots columns, all dim of them and the whole
  // spectrum when collapsing), and the last iteration's leading columns
  // S_prev, prev_rows x n_roots: X_prev = V S_prev, because V only grows
  // between collapses. prev_rows is 0 before the first. y_host and m_host hold
  // the collapse's uploads (Y's columns past X, and their block of Y^T H Y)
  // until the stream has read them.
  std::vector<T> s_now(keep_previous ? columns * columns : (track_previous ? columns * roots : 0));
  std::vector<T> spectrum(keep_previous ? columns : 0);
  std::vector<T> s_prev(track_previous ? columns * roots : 0);
  int32_t prev_rows = 0;
  std::vector<T> y_host(keep_previous ? columns * roots : 0);
  std::vector<T> m_host(keep_previous ? roots * roots : 0);
  // X_prev = V S_prev into previous_eigenvectors, or X itself before there is
  // a previous iteration. Valid until a collapse rewrites V or S_prev moves
  // on, so a collapsing iteration writes it first.
  const auto write_previous = [&]() -> Status {
    if (previous_eigenvectors == nullptr) {
      return {};
    }
    if (prev_rows == 0) {
      return gpu_status(wwr::wwrMemcpyAsync(previous_eigenvectors, eigenvectors, block_bytes,
                                            wwr::wwrMemcpyDeviceToDevice, stream));
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(projection.data(), s_prev.data(),
                                           static_cast<size_t>(prev_rows) * roots * sizeof(T),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(gemm(blas, n, n_roots, prev_rows, &kOne<T>, v.data(), projection.data(),
                            &kZero<T>, previous_eigenvectors)));
    // s_prev may change on the host before the stream reads it.
    return res.sync();
  };
  DavidsonInfo info;
  int32_t dim = guess_count;
  int32_t filled = 0; // columns of av already holding A v
  for (int32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    info.iterations = iteration;

    // 1. Extend A V over the new columns only: one apply per iteration. V^T A V
    //    gains the new columns' rows, V_new^T (A V), which with the rows before
    //    them hold every element of its lower triangle: the old rows are
    //    never recomputed. The new diagonal elements are then rewritten by
    //    dot, in device pointer mode, so with no sync: v^T A v sums n terms of
    //    one sign, which rocBLAS's gemm of this shape on the gfx1200 gets up
    //    to ~1e-12 relative wrong (its dot, and cuBLAS's gemm, near 1e-15). That
    //    error is the Ritz value's, a collapse writes it back as diag(theta),
    //    and the residual then floors at |theta| times it (see the README).
    if (dim > filled) {
      const size_t offset = static_cast<size_t>(filled) * nz;
      FBF_TRY(op.apply(res, dim - filled, v.data() + offset, av.data() + offset));
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(
          blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim - filled, dim, n, &kOne<T>,
          v.data() + offset, n, av.data(), n, &kZero<T>, vav.data() + filled, max_subspace)));
      {
        const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                                CreatePolicy<wwr::wwrblasStatus_t>>
            device_result{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
        for (auto j = static_cast<size_t>(filled); j < static_cast<size_t>(dim); ++j) {
          FBF_TRY(
              gpu_status(wwr::dot<T, int32_t>(blas, n, v.data() + (j * nz), 1, av.data() + (j * nz),
                                              1, vav.data() + (j * (columns + 1)))));
        }
      }
      filled = dim;
    }

    // 2. Rayleigh-Ritz: H = V^T A V packed at dim, then H = S diag(ritz) S^T,
    //    which syncs; theta comes back while the stream is idle, and for
    //    GD+k S's leading n_roots columns, or before a collapse all of S and
    //    the spectrum.
    const bool collapse = dim + n_roots > max_subspace;
    const bool whole = collapse && keep_previous && prev_rows > 0;
    const size_t dim_bytes = static_cast<size_t>(dim) * sizeof(T);
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(h.data(), dim_bytes, vav.data(), columns * sizeof(T),
                                             dim_bytes, static_cast<size_t>(dim),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(diagonalize<T>(res, dim, h.data(), ritz.data()));
    // ||H||_2 is max |theta| over the ascending spectrum: one of its two ends.
    T theta_top{};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta.data(), ritz.data(), roots * sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&theta_top, ritz.data() + (dim - 1), sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    if (track_previous) {
      const size_t s_columns = whole ? static_cast<size_t>(dim) : roots;
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s_now.data(), h.data(),
                                             static_cast<size_t>(dim) * s_columns * sizeof(T),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
    }
    if (whole) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(spectrum.data(), ritz.data(), dim_bytes,
                                             wwr::wwrMemcpyDeviceToHost, stream)));
    }
    FBF_TRY(res.sync());
    const T h_norm = std::max(std::abs(theta.front()), std::abs(theta_top));

    // 3. Ritz vectors X = V S and their images A X = (A V) S, from the leading
    //    n_roots columns of S: A is never applied to X.
    FBF_TRY(gpu_status(
        gemm(blas, n, n_roots, dim, &kOne<T>, v.data(), h.data(), &kZero<T>, eigenvectors)));
    FBF_TRY(gpu_status(
        gemm(blas, n, n_roots, dim, &kOne<T>, av.data(), h.data(), &kZero<T>, ritz_av.data())));

    // 4. Residuals R = A X - X diag(theta), and the relative test per root.
    //    The norms land on the device and come back together, one sync.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(residual.data(), ritz_av.data(), block_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    for (size_t i = 0; i < roots; ++i) {
      const T neg_theta = -theta[i];
      FBF_TRY(gpu_status(wwr::axpy<T, int32_t>(blas, n, &neg_theta, eigenvectors + (i * nz), 1,
                                               residual.data() + (i * nz), 1)));
    }
    {
      const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                              CreatePolicy<wwr::wwrblasStatus_t>>
          device_result{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
      for (size_t i = 0; i < roots; ++i) {
        FBF_TRY(gpu_status(
            wwr::nrm2<T, int32_t>(blas, n, residual.data() + (i * nz), 1, norms.data() + i)));
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(residual_norms.data(), norms.data(), roots * sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    bool all_converged = true;
    T max_norm{};
    for (size_t i = 0; i < roots; ++i) {
      const T norm = residual_norms[i];
      converged[i] = norm <= tolerance * std::max(std::abs(theta[i]), h_norm);
      all_converged = all_converged && converged[i];
      max_norm = std::max(max_norm, norm);
    }
    info.max_residual_norm = static_cast<double>(max_norm);
    FBF_LOG_DEBUG("davidson_solve: iteration {}, subspace {}, max residual {:.3e}", iteration, dim,
                  info.max_residual_norm);
    if (all_converged) {
      FBF_TRY(write_eigenvalues());
      FBF_TRY(write_previous());
      return info;
    }

    // 5. Corrections for the whole block, from the residuals and the Ritz
    //    vectors; a converged root's is dropped below.
    FBF_TRY(precondition(res, std::span<const T>{theta}, eigenvectors, residual.data(),
                         correction.data()));

    // 6. Collapse if a full new block would not fit: V onto span[X, X_prev]
    //    = V Y, Y = [S, S_prev padded with zero rows] orthonormalized (GD+k).
    //    X = V S and A X are already formed, so only Y's other columns Y_2 cost
    //    a gemm each, into the residual block, free once the corrections are
    //    made. In S's eigenbasis S is the leading unit vectors and H is
    //    diag(spectrum), so Y_2 = S_full Q with Q zero on S's rows, and
    //    V^T A V becomes diag(theta) beside Q^T diag(spectrum) Q.
    //    max_subspace >= 2 * n_roots leaves room for a block after X alone,
    //    3 * n_roots after X and X_prev.
    if (collapse) {
      // X_prev's V and S_prev are about to go.
      FBF_TRY(write_previous());
      int32_t extra = 0; // Y_2's columns, at most n_roots
      if (whole) {
        const auto d = static_cast<size_t>(dim);
        const auto rows = static_cast<size_t>(prev_rows);
        std::vector<double> q(d * roots, 0.0);
        for (size_t c = 0; c < roots; ++c) {
          // S_full^T S_prev's column c off S's rows, then off the Q before it.
          double *const p = q.data() + (static_cast<size_t>(extra) * d);
          for (size_t r = roots; r < d; ++r) {
            double dot = 0.0;
            for (size_t i = 0; i < rows; ++i) {
              dot += static_cast<double>(s_now[i + (r * d)]) *
                     static_cast<double>(s_prev[i + (c * rows)]);
            }
            p[r] = dot;
          }
          for (int32_t pass = 0; pass < 2; ++pass) {
            for (int32_t a = 0; a < extra; ++a) {
              const double *const qa = q.data() + (static_cast<size_t>(a) * d);
              double overlap = 0.0;
              for (size_t r = roots; r < d; ++r) {
                overlap += qa[r] * p[r];
              }
              for (size_t r = roots; r < d; ++r) {
                p[r] -= overlap * qa[r];
              }
            }
          }
          double norm2 = 0.0;
          for (size_t r = roots; r < d; ++r) {
            norm2 += p[r] * p[r];
          }
          const double norm = std::sqrt(norm2);
          if (norm <= static_cast<double>(floor)) {
            std::fill_n(p, d, 0.0); // nearly in span X: dropped
            continue;
          }
          for (size_t r = roots; r < d; ++r) {
            p[r] /= norm;
          }
          ++extra;
        }
        const auto kept_columns = static_cast<size_t>(extra);
        for (size_t j = 0; j < kept_columns; ++j) {
          const double *const qj = q.data() + (j * d);
          for (size_t i = 0; i < d; ++i) {
            double value = 0.0;
            for (size_t r = roots; r < d; ++r) {
              value += static_cast<double>(s_now[i + (r * d)]) * qj[r];
            }
            y_host[i + (j * d)] = static_cast<T>(value);
          }
          for (size_t a = 0; a < kept_columns; ++a) {
            const double *const qa = q.data() + (a * d);
            double value = 0.0;
            for (size_t r = roots; r < d; ++r) {
              value += static_cast<double>(spectrum[r]) * qa[r] * qj[r];
            }
            m_host[a + (j * kept_columns)] = static_cast<T>(value);
          }
        }
      }
      const size_t extra_bytes = nz * static_cast<size_t>(extra) * sizeof(T);
      if (extra > 0) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(projection.data(), y_host.data(),
                                               static_cast<size_t>(dim * extra) * sizeof(T),
                                               wwr::wwrMemcpyHostToDevice, stream)));
        FBF_TRY(gpu_status(gemm(blas, n, extra, dim, &kOne<T>, v.data(), projection.data(),
                                &kZero<T>, residual.data())));
      }
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), eigenvectors, block_bytes,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      if (extra > 0) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data() + (roots * nz), residual.data(),
                                               extra_bytes, wwr::wwrMemcpyDeviceToDevice, stream)));
        FBF_TRY(gpu_status(gemm(blas, n, extra, dim, &kOne<T>, av.data(), projection.data(),
                                &kZero<T>, residual.data())));
      }
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(av.data(), ritz_av.data(), block_bytes,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      if (extra > 0) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(av.data() + (roots * nz), residual.data(),
                                               extra_bytes, wwr::wwrMemcpyDeviceToDevice, stream)));
      }
      FBF_TRY(
          gpu_status(wwr::wwrMemsetAsync(vav.data(), 0, columns * columns * sizeof(T), stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(vav.data(), (columns + 1) * sizeof(T), ritz.data(),
                                               sizeof(T), sizeof(T), roots,
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
      if (extra > 0) {
        const size_t extra_row_bytes = static_cast<size_t>(extra) * sizeof(T);
        FBF_TRY(gpu_status(
            wwr::wwrMemcpy2DAsync(vav.data() + (roots * (columns + 1)), columns * sizeof(T),
                                  m_host.data(), extra_row_bytes, extra_row_bytes,
                                  static_cast<size_t>(extra), wwr::wwrMemcpyHostToDevice, stream)));
      }
      dim = n_roots + extra;
      filled = dim;
    }

    // 7. C -= V (V^T C), twice: one pass loses orthogonality once V is large.
    for (int32_t pass = 0; pass < 2; ++pass) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, dim, n_roots, n, &kOne<T>, v.data(),
                                                 correction.data(), &kZero<T>, projection.data())));
      FBF_TRY(gpu_status(gemm(blas, n, n_roots, dim, &kNegOne<T>, v.data(), projection.data(),
                              &kOne<T>, correction.data())));
    }

    // 8. The unconverged corrections among themselves, each survivor
    //    normalized and appended to V.
    int32_t kept = 0;
    if (options.canonical_orthonormalization) {
      // Appended to V and orthonormalized there as a block by
      // orthonormalize, which compacts the survivors to the front. A
      // correction at or below the floor, or a mode of them that is, adds
      // nothing and is dropped.
      int32_t candidates = 0;
      for (size_t i = 0; i < roots; ++i) {
        if (converged[i]) {
          continue;
        }
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
            v.data() + (static_cast<size_t>(dim + candidates) * nz), correction.data() + (i * nz),
            nz * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream)));
        ++candidates;
      }
      kept = FBF_TRY(
          orthonormalize<T>(res, n, candidates, v.data() + (static_cast<size_t>(dim) * nz), floor));
    } else {
      // Gram-Schmidt: a candidate is projected off the ones accepted before
      // it, contiguous in V, by two gemvs a pass and two passes, so the
      // overlaps stay on the device and only the norm comes back.
      for (size_t i = 0; i < roots; ++i) {
        if (converged[i]) {
          continue;
        }
        T *candidate = correction.data() + (i * nz);
        const T *const accepted = v.data() + (static_cast<size_t>(dim) * nz);
        for (int32_t pass = 0; kept > 0 && pass < 2; ++pass) {
          FBF_TRY(
              gpu_status(wwr::gemv<T, int32_t>(blas, wwr::WWRBLAS_OP_T, n, kept, &kOne<T>, accepted,
                                               n, candidate, 1, &kZero<T>, overlaps.data(), 1)));
          FBF_TRY(gpu_status(wwr::gemv<T, int32_t>(blas, wwr::WWRBLAS_OP_N, n, kept, &kNegOne<T>,
                                                   accepted, n, overlaps.data(), 1, &kOne<T>,
                                                   candidate, 1)));
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
    }
    if (kept == 0) {
      FBF_TRY(write_eigenvalues());
      if (!collapse) {
        FBF_TRY(write_previous());
      }
      return err_divergence("davidson_solve: no correction survived re-orthogonalization at "
                            "iteration {}, so the subspace cannot grow (max residual {:.3e})",
                            iteration, info.max_residual_norm);
    }
    if (iteration == options.max_iterations) {
      if (!collapse) {
        FBF_TRY(write_previous());
      }
      break;
    }

    // X is X_prev to the next iteration: S's leading columns, or after a
    // collapse the leading unit vectors of the new V.
    if (track_previous) {
      if (collapse) {
        std::fill(s_prev.begin(), s_prev.end(), T{0});
        for (size_t i = 0; i < roots; ++i) {
          s_prev[i + (i * roots)] = T{1};
        }
        prev_rows = n_roots;
      } else {
        std::copy_n(s_now.begin(), static_cast<size_t>(dim) * roots, s_prev.begin());
        prev_rows = dim;
      }
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
