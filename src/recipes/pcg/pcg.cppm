/// @file
/// @brief Block preconditioned conjugate gradients on A X = B for a symmetric
/// positive definite A known only as a LinearOperator, on the GPU over wwr's
/// BLAS.
///
/// The README has the recurrence, the stopping rules and why none of the
/// buffers is the workspace's.

export module free_breakfast.recipes.pcg;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // BLAS, wwrMemcpyAsync, ScopedPointerMode, CreatePolicy
import free_breakfast.core.resources;
export import free_breakfast.recipes.linear_operator;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The knobs of one pcg_solve call.
struct PcgOptions {
  /// @brief Column j has converged when ||r_j||_2 <= its tolerance: absolute,
  /// on the residual of the system as given. A caller wanting a relative rule
  /// passes rel * ||b_j|| per column in @ref tolerances.
  double tolerance = 1e-10;
  /// @brief When not empty, one tolerance per column, replacing
  /// @ref tolerance. Read during the call only.
  std::span<const double> tolerances;
  /// @brief When not empty, ||b_j||_2 per column as the caller already holds
  /// it (to set its tolerances, say), taken as the starting residual's norm
  /// instead of the call's own nrm2 of each column. It is the recurrence's
  /// ||r_0|| (unpreconditioned, r_0.r_0 is its square), so it should be that
  /// nrm2. From X = 0 only; read during the call only.
  std::span<const double> norms;
  /// @brief Sweeps before a column still above its tolerance stops. At least 1.
  int32_t max_iterations = 100;
  /// @brief Start from the X passed in, R = B - A X by one product over every
  /// column, instead of from X = 0, R = B.
  bool start_from_x = false;
  /// @brief A direction with p.Ap <= 0 is Divergence. Off, that column stops
  /// there instead, its X as the sweep before left it, and is flagged in
  /// PcgInfo::converged.
  bool check_curvature = true;
  /// @brief Columns still above their tolerance after max_iterations stop
  /// there and are flagged in PcgInfo::converged, and the call returns
  /// normally, instead of MaxIterationsExceeded.
  bool stop_silently = false;
  /// @brief Report each column's true residual ||A x_j - b_j||, from one more
  /// product over every column at the end.
  bool true_residuals = true;
};

/// @brief What a pcg_solve that returned normally did.
struct PcgInfo {
  /// @brief Lockstep sweeps run: the most products any one column took.
  int32_t sweeps = 0;
  /// @brief Per column, the products of A it took: the sweeps it was active.
  std::vector<int32_t> iterations;
  /// @brief Per column, true when its recurrence residual reached its
  /// tolerance; false only for a column stopped by stop_silently or, with
  /// check_curvature off, by a non-positive curvature.
  std::vector<bool> converged;
  /// @brief Per column, ||A x_j - b_j||_2 from the final product, when
  /// true_residuals; empty otherwise.
  std::vector<double> residuals;
};

/// @brief Solve A X = B for the @p k columns of @p b by preconditioned
/// conjugate gradients, every column in lockstep.
///
/// Each sweep gathers the columns still above their tolerance into one
/// contiguous block, applies @p op to it once, and steps those columns alone:
/// a column that has stopped is never read or written again. The
/// preconditioner M ~ A^-1 is applied the same way, to the columns the sweep
/// stepped; with @p precondition nullptr it is the identity, and the search
/// direction is the residual itself (plain CG, r.r taken as ||r||^2).
///
/// @p op is applied as op.apply(res, count, x, y), a LinearOperator; or, when
/// it has one, as op.apply(res, columns, x, y), where @p columns
/// (std::span<const size_t>) lists the original index of each of the block's
/// columns, ascending: for an operator that differs per column, such as a
/// shift per right-hand side.
///
/// Everything is queued on res.stream(), @p op's and @p precondition's
/// applies in host pointer mode on res.blas(); a sweep's reductions are
/// formed in device pointer mode and read back in two copies, each a sync.
/// The residual and direction blocks are pool buffers of the call's own,
/// allocated on entry, so @p op and @p precondition may take
/// res.device_workspace(). The device res.dev_idx() names is made current for
/// the call. X's last update is ordered on res.stream(), not synced.
///
/// @param op            Applied as above. Must be symmetric positive definite
///                      on the columns' span.
/// @param precondition  A LinearOperator applying M, symmetric positive
///                      definite, or nullptr for none.
/// @param n             The order of @p op.
/// @param k             Right-hand sides, at least 1.
/// @param b             Device, n x k, leading dimension n.
/// @param x             In/out: device, n x k, leading dimension n. Read only
///                      with start_from_x.
/// @return Per column, the products and whether it converged, and the true
///         residuals unless switched off. Otherwise InvalidConfig for a shape
///         or option outside the bounds above; MaxIterationsExceeded when a
///         column is still above its tolerance after max_iterations (unless
///         stop_silently); Divergence when a direction has p.Ap <= 0 (unless
///         check_curvature is off); or the first failing call's error, @p op's
///         and @p precondition's included. After an error, @p x holds the
///         iterate as far as it got.
template <real_fp T, class Op, class Precondition>
  requires std::same_as<std::remove_cvref_t<Precondition>, std::nullptr_t> ||
           LinearOperator<std::remove_reference_t<Precondition>, T>
[[nodiscard]] Result<PcgInfo> pcg_solve(DeviceResources &res, Op &op, Precondition &&precondition,
                                        const int32_t n, const int32_t k, const T *b, T *x,
                                        const PcgOptions &options = {}) {
  constexpr bool kPreconditioned = !std::same_as<std::remove_cvref_t<Precondition>, std::nullptr_t>;
  constexpr bool kColumns = requires(std::span<const size_t> columns, const T *from, T *to) {
    { op.apply(res, columns, from, to) } -> std::convertible_to<Status>;
  };
  static_assert(kColumns || LinearOperator<Op, T>,
                "pcg_solve: op needs apply(res, count, x, y) or apply(res, columns, x, y)");
  if (n < 1 || k < 1 || options.max_iterations < 1 ||
      (!options.tolerances.empty() && std::cmp_not_equal(options.tolerances.size(), k)) ||
      (!options.norms.empty() &&
       (options.start_from_x || std::cmp_not_equal(options.norms.size(), k)))) {
    return err_config("pcg_solve: needs n >= 1, k >= 1, max_iterations >= 1, tolerances empty "
                      "or one per column, and norms empty or one per column from X = 0 (n = {}, "
                      "k = {}, max_iterations = {}, tolerances = {}, norms = {})",
                      n, k, options.max_iterations, options.tolerances.size(),
                      options.norms.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nz = static_cast<size_t>(n);
  const auto cols = static_cast<size_t>(k);
  const size_t block = nz * cols;
  const auto tolerance = [&](const size_t j) {
    return static_cast<T>(options.tolerances.empty() ? options.tolerance : options.tolerances[j]);
  };

  // Every scalar a BLAS call takes is a host value, so host pointer mode for
  // the call, the previous mode restored; reduce below switches to device mode
  // only while it queues the reductions it reads back.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};

  // R, P, the gathered block an apply reads and the block it writes, and the
  // 2 k reductions a readback copies: held across op.apply and
  // precondition.apply, each of which may take res.device_workspace(), so the
  // call's own (docs/style.md "Scratch memory"). Their frees are ordered on
  // the stream after the work.
  Buffers::device<T> vectors((4 * block) + (2 * cols), res.device());
  T *const r = vectors.data();
  T *const p = r + block;
  T *const xs = p + block;
  T *const ys = xs + block;
  T *const scalars = ys + block;

  // The reductions @p issue queues, each into its slot of scalars, in device
  // pointer mode, then the first @p count read back into read by one copy and
  // one sync: a sweep's dots and norms cost one round trip, not one each. The
  // sums are the host-mode calls' own.
  std::vector<T> read(2 * cols);
  const auto reduce = [&](const size_t count, const auto &issue) -> Status {
    FBF_TRY(gpu_status(wwr::wwrblasSetPointerMode(blas, wwr::WWRBLAS_POINTER_MODE_DEVICE)));
    const Status issued = issue();
    FBF_TRY(gpu_status(wwr::wwrblasSetPointerMode(blas, wwr::WWRBLAS_POINTER_MODE_HOST)));
    FBF_TRY(issued);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(read.data(), scalars, count * sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    return res.sync();
  };
  const auto dot = [&](const T *u, const T *v, T *into) {
    return gpu_status(wwr::dot<T, int32_t>(blas, n, u, 1, v, 1, into));
  };
  const auto norm = [&](const T *u, T *into) {
    return gpu_status(wwr::nrm2<T, int32_t>(blas, n, u, 1, into));
  };
  // @p op over the block @p from of the listed columns' vectors, into @p to.
  const auto apply = [&](const std::vector<size_t> &columns, const T *from, T *to) -> Status {
    if constexpr (kColumns) {
      return op.apply(res, std::span<const size_t>{columns}, from, to);
    } else {
      return op.apply(res, static_cast<int32_t>(columns.size()), from, to);
    }
  };
  // The listed columns of @p from packed into xs, or @p from itself when the
  // list is every column (it is ascending, so that is the identity gather).
  const auto gather = [&](const std::vector<size_t> &columns, const T *from) -> Result<const T *> {
    if (columns.size() == cols) {
      return from;
    }
    for (size_t i = 0; i < columns.size(); ++i) {
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(xs + (i * nz), from + (columns[i] * nz), nz * sizeof(T),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
    }
    return xs;
  };
  std::vector<size_t> every(cols);
  std::iota(every.begin(), every.end(), size_t{0});

  // R = B - A X0 or B, and P = S = M R over every column.
  if (options.start_from_x) {
    FBF_TRY(apply(every, x, r));
    for (size_t j = 0; j < cols; ++j) {
      FBF_TRY(gpu_status(wwr::scal<T, int32_t>(blas, n, &kNegOne<T>, r + (j * nz), 1)));
      FBF_TRY(
          gpu_status(wwr::axpy<T, int32_t>(blas, n, &kOne<T>, b + (j * nz), 1, r + (j * nz), 1)));
    }
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(x, 0, block * sizeof(T), stream)));
    FBF_TRY(gpu_status(
        wwr::wwrMemcpyAsync(r, b, block * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  if constexpr (kPreconditioned) {
    FBF_TRY(precondition.apply(res, k, r, p));
  } else {
    FBF_TRY(gpu_status(
        wwr::wwrMemcpyAsync(p, r, block * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  // ||r|| (unless the caller's norms give it) and, preconditioned, r.s per
  // column.
  std::vector<T> rs(cols);
  std::vector<T> r_norm(cols);
  const bool own_norms = options.norms.empty();
  const size_t at_rs = own_norms ? cols : 0;
  if (own_norms || kPreconditioned) {
    FBF_TRY(reduce(at_rs + (kPreconditioned ? cols : 0), [&]() -> Status {
      for (size_t j = 0; j < cols; ++j) {
        if (own_norms) {
          FBF_TRY(norm(r + (j * nz), scalars + j));
        }
        if constexpr (kPreconditioned) {
          FBF_TRY(dot(r + (j * nz), p + (j * nz), scalars + at_rs + j));
        }
      }
      return {};
    }));
  }
  for (size_t j = 0; j < cols; ++j) {
    r_norm[j] = own_norms ? read[j] : static_cast<T>(options.norms[j]);
    if constexpr (kPreconditioned) {
      rs[j] = read[at_rs + j];
    } else {
      rs[j] = r_norm[j] * r_norm[j];
    }
  }

  PcgInfo info;
  info.iterations.assign(cols, 0);
  std::vector<bool> stopped(cols); // by a non-positive curvature, check off
  std::vector<size_t> active;      // above tolerance at the sweep's start
  std::vector<size_t> stepped;     // of those, the ones the sweep moved
  for (;;) {
    // A NaN norm is not at or below tolerance, so its column stays active and
    // its NaN curvature then stops it.
    active.clear();
    T largest{};
    for (size_t j = 0; j < cols; ++j) {
      if (!stopped[j] && !(r_norm[j] <= tolerance(j))) {
        active.push_back(j);
        largest = std::max(largest, r_norm[j]);
      }
    }
    if (active.empty()) {
      break;
    }
    if (info.sweeps == options.max_iterations) {
      if (!options.stop_silently) {
        return err_max_iterations_exceeded(
            "pcg_solve: {} of {} columns above tolerance after {} sweeps, the largest ||r|| {:.3e}",
            active.size(), cols, options.max_iterations, static_cast<double>(largest));
      }
      break;
    }
    ++info.sweeps;
    FBF_LOG_DEBUG("pcg_solve: sweep {}, {} of {} columns active, largest ||r|| {:.3e}", info.sweeps,
                  active.size(), cols, static_cast<double>(largest));

    // A P over the active columns, one apply into YS, and every p.Ap read
    // back at once.
    const T *const p_active = FBF_TRY(gather(active, p));
    FBF_TRY(apply(active, p_active, ys));
    FBF_TRY(reduce(active.size(), [&]() -> Status {
      for (size_t i = 0; i < active.size(); ++i) {
        FBF_TRY(dot(p + (active[i] * nz), ys + (i * nz), scalars + i));
      }
      return {};
    }));
    stepped.clear();
    for (size_t i = 0; i < active.size(); ++i) {
      const size_t j = active[i];
      const size_t at = j * nz;
      ++info.iterations[j];
      const T pap = read[i];
      if (!(pap > T{0})) {
        if (options.check_curvature) {
          return err_divergence("pcg_solve: p.Ap = {:.3e} in column {} after {} products: A is "
                                "not positive definite",
                                static_cast<double>(pap), j, info.iterations[j]);
        }
        stopped[j] = true;
        continue;
      }
      const T alpha = rs[j] / pap;
      const T neg_alpha = -alpha;
      FBF_TRY(gpu_status(wwr::axpy<T, int32_t>(blas, n, &alpha, p + at, 1, x + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<T, int32_t>(blas, n, &neg_alpha, ys + (i * nz), 1, r + at, 1)));
      stepped.push_back(j);
    }
    if (stepped.empty()) {
      continue;
    }

    // S = M R over the stepped columns, into YS (A P is consumed), then their
    // ||r|| and r.s read back at once. Without M, S is R and r.s is ||r||^2.
    const size_t m = stepped.size();
    if constexpr (kPreconditioned) {
      const T *const r_stepped = FBF_TRY(gather(stepped, r));
      FBF_TRY(precondition.apply(res, static_cast<int32_t>(m), r_stepped, ys));
    }
    FBF_TRY(reduce(kPreconditioned ? 2 * m : m, [&]() -> Status {
      for (size_t i = 0; i < m; ++i) {
        FBF_TRY(norm(r + (stepped[i] * nz), scalars + i));
        if constexpr (kPreconditioned) {
          FBF_TRY(dot(r + (stepped[i] * nz), ys + (i * nz), scalars + m + i));
        }
      }
      return {};
    }));
    // P = S + beta P, for the columns another sweep can read: not one that
    // has reached its tolerance, nor any after the last sweep.
    const bool last = info.sweeps == options.max_iterations;
    for (size_t i = 0; i < m; ++i) {
      const size_t j = stepped[i];
      r_norm[j] = read[i];
      if (last || r_norm[j] <= tolerance(j)) {
        continue;
      }
      const size_t at = j * nz;
      T rs_new{};
      const T *s = r + at;
      if constexpr (kPreconditioned) {
        s = ys + (i * nz);
        rs_new = read[m + i];
      } else {
        rs_new = r_norm[j] * r_norm[j];
      }
      const T beta = rs_new / rs[j];
      FBF_TRY(gpu_status(wwr::scal<T, int32_t>(blas, n, &beta, p + at, 1)));
      FBF_TRY(gpu_status(wwr::axpy<T, int32_t>(blas, n, &kOne<T>, s, 1, p + at, 1)));
      rs[j] = rs_new;
    }
  }

  info.converged.resize(cols);
  for (size_t j = 0; j < cols; ++j) {
    info.converged[j] = !stopped[j] && r_norm[j] <= tolerance(j);
  }
  // The true residuals A X - B from one more product over every column: the
  // recurrence's R drifts from them in rounding.
  if (options.true_residuals) {
    FBF_TRY(apply(every, x, ys));
    for (size_t j = 0; j < cols; ++j) {
      FBF_TRY(gpu_status(
          wwr::axpy<T, int32_t>(blas, n, &kNegOne<T>, b + (j * nz), 1, ys + (j * nz), 1)));
    }
    FBF_TRY(reduce(cols, [&]() -> Status {
      for (size_t j = 0; j < cols; ++j) {
        FBF_TRY(norm(ys + (j * nz), scalars + j));
      }
      return {};
    }));
    info.residuals.resize(cols);
    for (size_t j = 0; j < cols; ++j) {
      info.residuals[j] = static_cast<double>(read[j]);
    }
  }
  return info;
}

} // namespace free_breakfast
