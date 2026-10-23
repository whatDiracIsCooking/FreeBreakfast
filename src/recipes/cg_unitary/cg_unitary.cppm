/// @file
/// @brief Riemannian conjugate gradient on the unitary group U(n) (the
/// orthogonal group O(n) for a real T), after Abrudan, Eriksson & Koivunen,
/// "Conjugate gradient algorithm for optimization under unitary matrix
/// constraint", Signal Processing 89 (2009) 1704-1714.
///
/// Minimize or maximize a real J(W) subject to W^H W = I, by conjugate
/// gradient along the geodesics exp(t H) W, with the step chosen by one of the
/// paper's two almost-periodicity line searches. The README has the
/// iteration, the searches and what was changed from calaman's port.

module;

// kCgMaxSamples, which the sample buffers here and the kernels' arrays share.
// GMF-included, as src/core/kernel/kernel.cppm does block_params.h, so the
// constant is written once. Not the bridge: see this header's comment.
#include "cg_unitary_limits.h"

export module free_breakfast.recipes.cg_unitary;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kPi
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // gemm, geam, dot, dotc, nrm2, axpy, scal, ScopedPointerMode
import free_breakfast.core.resources;
import free_breakfast.recipes.expm;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief Whether cg_unitary descends or ascends J.
///
/// The paper minimizes, and maximizes by its Remark 1, which flips the sign of
/// the step and of the derivative samples and nothing else.
enum class CgDirection {
  Minimize, ///< Drive J down
  Maximize, ///< Drive J up
};

/// @brief Which of the paper's line searches picks the step.
enum class LineSearchMethod {
  /// Table 1: a degree-P polynomial through the derivative over one period
  /// T_mu, taking its first zero crossing. P + 1 samples, one of them free.
  Polynomial,
  /// Table 2: a Hann-windowed DFT of the derivative over several periods,
  /// taking the best of the minima it finds. Several times the samples, for
  /// larger steps early; falls back to Table 1 when it finds none past its
  /// first sample interval, which near the optimum is every search.
  Dft,
};

/// @brief What cg_unitary needs of a real J on U(n).
///
/// - `F::kOrder`, the paper's q (section 3.1): the highest power of t in
///   J(W + t Z). It fixes the highest frequency of J along a geodesic, so the
///   search interval T_mu = 2 pi / (q |omega_max|). It must be right: too
///   small and T_mu overshoots, breaking the at-most-one-cycle argument both
///   searches rest on. tr(W^H R W N) has q = 2; the JADE criterion has q = 4.
/// - `f.euclidean_gradient(res, n, w, psi)`: Psi = dJ/dW* at @p w, the
///   gradient with dJ = 2 Re tr(Psi^H dW). The Riemannian gradient is built
///   from it by the solver.
/// - `f.value(res, n, w, psi, out)`: J(w) into the DEVICE scalar @p out, of
///   which only the real part is read. Device, because Table 2 compares a
///   window of them without a host round trip. @p psi is the gradient at @p w,
///   already computed, for a J that is an inner product with it (most are).
///
/// Every matrix is an n x n device pointer with packed leading dimension n.
/// Both calls are queued on res.stream() with res.blas() in host pointer
/// mode, and must leave it so. Either may take res.device_workspace(): no
/// argument points into it, and the solver keeps nothing there across them.
template <class F, class T>
concept UnitaryCostFunction =
    wwr::usual_fp<T> &&
    requires(const F &f, DeviceResources &res, int32_t n, const T *w, const T *psi, T *out) {
      { F::kOrder } -> std::convertible_to<int32_t>;
      { f.euclidean_gradient(res, n, w, out) } -> std::convertible_to<Status>;
      { f.value(res, n, w, psi, out) } -> std::convertible_to<Status>;
    };

/// @brief The knobs of one cg_unitary call. The defaults are the paper's
/// (section 5.1) except where noted.
template <wwr::real_fp R>
struct CgOptions {
  /// @brief Steps before giving up. At least 1.
  int32_t max_iterations = 300;
  /// @brief Converged when <G, G> = 0.5 ||G||_F^2 falls below this. The
  /// gradient, not the change in J: along a geodesic J is almost periodic, so
  /// a short step is no evidence of a small gradient.
  R gradient_tolerance = std::same_as<R, float> ? R{1e-10} : R{1e-20};
  /// @brief The line search.
  LineSearchMethod method = LineSearchMethod::Polynomial;
  /// @brief P, Table 1's polynomial degree, 1 to 5.
  int32_t polynomial_order = 5;
  /// @brief K, Table 2's samples per period T_mu, at least 2 (Nyquist).
  ///
  /// 5, not the paper's 3. Its N_DFT is unbounded, but the samples are capped
  /// at kCgMaxSamples - 1 = 31, and at the cap K and N_T trade against each
  /// other: K = 3 spends them on ten periods and stalls around -19 dB of
  /// diagonality on the Brockett criterion; K = 5 spends them on six and
  /// reaches the maximum.
  int32_t dft_samples_per_period = 5;
  /// @brief N_T, the periods Table 2's window spans before the cap. At least 1.
  int32_t dft_periods = 10;
  /// @brief Steps between resets of the direction to the gradient; 0 is the
  /// paper's n^2, the real dimension of U(n).
  int32_t reset_period = 0;
};

/// @brief What a converged cg_unitary did.
struct CgInfo {
  /// @brief Steps taken.
  int32_t iterations = 0;
  /// @brief Times the direction was set to the gradient, the first included.
  int32_t resets = 0;
  /// @brief Searches along a conjugate direction that found no step, each
  /// retried along the gradient.
  int32_t line_search_failures = 0;
  /// @brief The final <G, G>.
  double gradient_norm_sq = 0.0;
  /// @brief The last step mu.
  double last_step = 0.0;
};

} // namespace free_breakfast

namespace free_breakfast {

// A real scalar as an element of T: complex types take a zero imaginary part.
template <wwr::usual_fp T>
T cg_scalar(const wwr::ComplexToRealType<T> x) {
  if constexpr (wwr::complex_fp<T>) {
    return T{x, 0};
  } else {
    return x;
  }
}

// Re tr(A^H B) over @p count packed elements, on the host. For a complex T it
// is the real dot of the two arrays read as 2 * count reals.
template <wwr::usual_fp T>
Result<wwr::ComplexToRealType<T>> frobenius_dot(const wwr::wwrblasHandle_t blas,
                                                const int64_t count, const T *const a,
                                                const T *const b) {
  using R = wwr::ComplexToRealType<T>;
  R out{};
  if constexpr (wwr::real_fp<T>) {
    FBF_TRY(gpu_status(wwr::dot<T, int64_t>(blas, count, a, 1, b, 1, &out)));
  } else {
    FBF_TRY(gpu_status(wwr::dot<R, int64_t>(blas, 2 * count, reinterpret_cast<const R *>(a), 1,
                                            reinterpret_cast<const R *>(b), 1, &out)));
  }
  return out;
}

// The call's own buffers. Every one is live across a cost callback, which may
// take res.device_workspace(), so none can be the workspace's; skew_eig and
// skew_expm also refuse arguments in it.
template <wwr::usual_fp T>
struct CgState {
  using R = wwr::ComplexToRealType<T>;
  using Z = wwr::RealToComplexType<R>;

  CgState(DeviceResources &res, const int32_t n)
      : nn(static_cast<size_t>(n) * static_cast<size_t>(n)), psi(nn, res.device()),
        grad(nn, res.device()), grad_next(nn, res.device()), dir(nn, res.device()),
        tmp(nn, res.device()), trial(nn, res.device()), psi_trial(nn, res.device()),
        rotation(nn, res.device()), eigvec(nn, res.device()),
        eigval(static_cast<size_t>(n), res.device()), dots(kCgMaxSamples, res.device()),
        cost_dots(kCgMaxSamples, res.device()) {}

  size_t nn;
  Buffers::device<T> psi;       // Euclidean gradient at W
  Buffers::device<T> grad;      // Riemannian gradient G at W
  Buffers::device<T> grad_next; // G at the new W
  Buffers::device<T> dir;       // the search direction H
  Buffers::device<T> tmp;       // Psi W^H; H W_i; G' - G
  Buffers::device<T> trial;     // a point exp(t H) W
  Buffers::device<T> psi_trial; // the Euclidean gradient there
  Buffers::device<T> rotation;  // exp(t H)
  Buffers::device<Z> eigvec;    // iH = U diag(w) U^H, from skew_eig
  Buffers::device<R> eigval;
  Buffers::device<T> dots;      // derivative samples, raw
  Buffers::device<T> cost_dots; // J samples, raw
};

// Table 1 steps 7-8 on the order + 1 samples in @p dots, spaced @p h: the
// first zero crossing in (0, order * h] of the interpolated derivative, or 0.
// Defined in cg_unitary.cpp, which holds every device launch; it takes
// res.device_workspace() and synchronizes once, to read the step back.
template <wwr::usual_fp T>
Result<wwr::ComplexToRealType<T>> polynomial_step(DeviceResources &res, const T *dots,
                                                  int32_t order, wwr::ComplexToRealType<T> h,
                                                  wwr::ComplexToRealType<T> sample_scale);

// Table 2 steps 8-13 on the @p num_samples samples in @p dots and @p cost_dots,
// spanning @p t_dft: the chosen minimum (maximum), or 0 when none is found. As
// polynomial_step for the workspace and the synchronization.
template <wwr::usual_fp T>
Result<wwr::ComplexToRealType<T>>
dft_step(DeviceResources &res, const T *dots, const T *cost_dots, int32_t num_samples,
         wwr::ComplexToRealType<T> sample_scale, wwr::ComplexToRealType<T> t_dft, bool maximizing);

// The step along the geodesic exp(sign mu H) W, H = s.dir, s.psi being the
// Euclidean gradient at @p w: mu, or 0 when no extremum was found. Leaves
// s.eigvec / s.eigval holding H's decomposition, for the step itself.
//
// One heevd per search, and every sample is exp(t_i H) evaluated directly from
// it: no matrix powers, so nothing accumulates. Sample 0 is @p w itself, with
// the gradient the solver already has.
template <wwr::usual_fp T, UnitaryCostFunction<T> CostF>
Result<wwr::ComplexToRealType<T>>
geodesic_search(DeviceResources &res, const int32_t n, const T *const w, const CostF &cost,
                const CgDirection direction, const CgOptions<wwr::ComplexToRealType<T>> &options,
                CgState<T> &s) {
  using R = wwr::ComplexToRealType<T>;
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nn = static_cast<int64_t>(s.nn);

  FBF_TRY(skew_eig<T>(res, n, s.dir.data(), s.eigvec.data(), s.eigval.data()));
  // |omega_max| = max |w|, one of the ascending spectrum's two ends: the
  // exact frequency bound, where calaman estimated one by power iteration.
  std::array<R, 2> ends{};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&ends[0], s.eigval.data(), sizeof(R),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&ends[1], s.eigval.data() + (n - 1), sizeof(R),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  const R omega = std::max(std::abs(ends[0]), std::abs(ends[1]));
  // A zero direction: nothing to search along.
  if (!(omega > R{0})) {
    return R{0};
  }
  const R t_mu = kTwo<R> * kPi<R> / (static_cast<R>(CostF::kOrder) * omega);
  const R sign = direction == CgDirection::Minimize ? R{-1} : R{1};
  // dJ/dmu = 2 sign Re tr(Psi^H H W_mu) at W_mu = exp(sign mu H) W.
  const R sample_scale = kTwo<R> * sign;

  // dots[i] = tr((H W_i)^H Psi_i) at mu_i = i h, and with want_cost
  // cost_dots[i] = J(W_i), all left on the device.
  const auto sample = [&](const int32_t count, const R h, const bool want_cost) -> Status {
    for (int32_t i = 0; i < count; ++i) {
      const T *trial = w;
      const T *psi = s.psi.data();
      if (i > 0) {
        FBF_TRY(skew_expm<T>(res, n, s.eigvec.data(), s.eigval.data(), sign * h * static_cast<R>(i),
                             s.rotation.data()));
        FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                 n, &kOne<T>, s.rotation.data(), n, w, n, &kZero<T>,
                                                 s.trial.data(), n)));
        FBF_TRY(cost.euclidean_gradient(res, n, s.trial.data(), s.psi_trial.data()));
        trial = s.trial.data();
        psi = s.psi_trial.data();
      }
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n,
                                               &kOne<T>, s.dir.data(), n, trial, n, &kZero<T>,
                                               s.tmp.data(), n)));
      {
        // Device pointer mode for this one reduction, so the sample stays on
        // the device; the guard puts host mode back.
        const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                                CreatePolicy<wwr::wwrblasStatus_t>>
            device_result{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
        if constexpr (wwr::real_fp<T>) {
          FBF_TRY(gpu_status(
              wwr::dot<T, int64_t>(blas, nn, s.tmp.data(), 1, psi, 1, s.dots.data() + i)));
        } else {
          FBF_TRY(gpu_status(
              wwr::dotc<T, int64_t>(blas, nn, s.tmp.data(), 1, psi, 1, s.dots.data() + i)));
        }
      }
      if (want_cost) {
        FBF_TRY(cost.value(res, n, trial, psi, s.cost_dots.data() + i));
      }
    }
    return {};
  };

  if (options.method == LineSearchMethod::Dft) {
    // N_DFT odd, at least 3, and inside the cap; the window then spans the
    // whole periods those samples cover at K per period, so a capped N_DFT
    // shortens the window rather than under-sampling it.
    const int32_t factor = options.dft_samples_per_period;
    const int32_t samples = std::clamp((2 * ((factor * options.dft_periods) / 2)) + 1, 3,
                                       (2 * ((kCgMaxSamples - 1) / 2)) + 1);
    const R t_dft = static_cast<R>(std::max(samples / factor, 1)) * t_mu;
    const R h = t_dft / static_cast<R>(samples);
    FBF_TRY(sample(samples, h, true));
    const R mu = FBF_TRY(dft_step<T>(res, s.dots.data(), s.cost_dots.data(), samples, sample_scale,
                                     t_dft, direction == CgDirection::Maximize));
    if (mu > h) {
      return mu;
    }
    // Table 1 instead, in two cases. No minimum survived: the paper sets
    // mu = 0 and stops (Table 2 step 11), which in a solver is a stall. Or the
    // one chosen lies inside the first sample interval, which the DFT cannot
    // place: the window has all but zeroed the samples there, and the
    // periodic reconstruction joins the window's two ends. Near the optimum
    // that is the usual case: H shrinks with G, so T_mu grows as 1 / |G|
    // while the first minimum stays O(1) away. Table 1's interpolant through
    // the same kind of spacing, unwindowed and anchored on the sample at 0,
    // resolves it. The DFT's advantage is large steps early, not local
    // accuracy.
  }

  const int32_t order = options.polynomial_order;
  const R h = t_mu / static_cast<R>(order);
  FBF_TRY(sample(order + 1, h, false));
  return polynomial_step<T>(res, s.dots.data(), order, h, sample_scale);
}

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief Minimize (maximize) @p cost over the unitary (orthogonal) matrices,
/// starting from and overwriting @p w. Paper Table 3.
///
/// Each step forms the Riemannian gradient G = Psi W^H - W Psi^H in u(n),
/// searches along the geodesic exp(sign mu H) W for the first extremum (Table
/// 1) or the best of several (Table 2), steps there, and updates the direction
/// H <- G' + gamma H with the approximate Polak-Ribiere gamma of eq. (10). H
/// is reset to G every reset_period steps, when <H, G> stops being positive
/// (Remark 2), and after a failed search, which is then retried along G.
///
/// Everything is queued on res.stream() with res.blas() held in host pointer
/// mode, and each step synchronizes the stream several times: inside each
/// heevd, to read the frequency bound and the step, and for the host-mode
/// reductions behind the convergence test and gamma. The iterate, gradients
/// and samples live in pool buffers of the call's own; the cost callbacks,
/// skew_eig and skew_expm take res.device_workspace(). The device
/// res.dev_idx() names is made current for the call.
///
/// @param res        Supplies the stream, the BLAS and solver handles and the
///                   scratch workspace.
/// @param n          Matrix order, at least 1.
/// @param w          In: the starting point W_0, unitary (orthogonal), n x n
///                   device, packed leading dimension n -- the identity for
///                   the paper's runs. Out: the last iterate, on every return
///                   after the first step.
/// @param cost       J; see UnitaryCostFunction.
/// @param direction  Minimize or maximize.
/// @param options    Tuning; see CgOptions.
/// @return The steps taken and the final gradient on convergence. Otherwise
///         InvalidConfig for an argument outside the bounds above;
///         MaxIterationsExceeded once max_iterations run out; Divergence when
///         no step is found even along the gradient; or the first failing
///         call's error, the cost callbacks' included.
template <wwr::usual_fp T, UnitaryCostFunction<T> CostF>
[[nodiscard]] Result<CgInfo> cg_unitary(DeviceResources &res, const int32_t n, T *const w,
                                        const CostF &cost, const CgDirection direction,
                                        const CgOptions<wwr::ComplexToRealType<T>> &options = {}) {
  using R = wwr::ComplexToRealType<T>;
  // !(tol >= 0) also rejects a NaN tolerance, which no gradient would meet.
  if (n < 1 || w == nullptr || options.max_iterations < 1 ||
      !(options.gradient_tolerance >= R{0}) || options.polynomial_order < 1 ||
      options.polynomial_order > 5 || options.dft_samples_per_period < 2 ||
      options.dft_periods < 1 || options.reset_period < 0) {
    return err_config("cg_unitary: needs n >= 1, a non-null w, max_iterations >= 1, "
                      "gradient_tolerance >= 0, 1 <= polynomial_order <= 5, "
                      "dft_samples_per_period >= 2, dft_periods >= 1 and reset_period >= 0 (n = "
                      "{}, max_iterations = {}, gradient_tolerance = {}, polynomial_order = {}, "
                      "dft_samples_per_period = {}, dft_periods = {}, reset_period = {})",
                      n, options.max_iterations, options.gradient_tolerance,
                      options.polynomial_order, options.dft_samples_per_period, options.dft_periods,
                      options.reset_period);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  // Every scalar is a host value and the reductions behind the control flow
  // write host results, so host pointer mode for the call; the previous mode
  // is restored.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};

  CgState<T> s{res, n};
  const auto nn = static_cast<int64_t>(s.nn);
  const size_t matrix_bytes = s.nn * sizeof(T);
  const auto adjoint = wwr::real_fp<T> ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_C;
  const R sign = direction == CgDirection::Minimize ? R{-1} : R{1};
  const int32_t reset_period = options.reset_period > 0 ? options.reset_period : n * n;

  // Paper steps 2 and 6: Psi = dJ/dW* at w into s.psi, then G = X - X^H with
  // X = Psi W^H, the projection onto u(n).
  const auto riemannian_gradient = [&](T *const g) -> Status {
    FBF_TRY(cost.euclidean_gradient(res, n, w, s.psi.data()));
    FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, adjoint, n, n, n, &kOne<T>,
                                             s.psi.data(), n, w, n, &kZero<T>, s.tmp.data(), n)));
    return gpu_status(wwr::geam<T, int32_t>(blas, wwr::WWRBLAS_OP_N, adjoint, n, n, &kOne<T>,
                                            s.tmp.data(), n, &kNegOne<T>, s.tmp.data(), n, g, n));
  };
  // <G, G> = 0.5 ||G||_F^2.
  const auto half_norm_sq = [&](const T *const g) -> Result<R> {
    R norm{};
    FBF_TRY(gpu_status(wwr::nrm2<T, int64_t>(blas, nn, g, 1, &norm)));
    return R{0.5} * norm * norm;
  };

  // G and G', swapped each step.
  T *grad = s.grad.data();
  T *grad_next = s.grad_next.data();

  CgInfo info;
  // Whether H is G itself, in which case a failed search has no steeper
  // direction to retry along.
  bool steepest = true;
  const auto reset = [&]() -> Status {
    ++info.resets;
    steepest = true;
    return gpu_status(wwr::wwrMemcpyAsync(s.dir.data(), grad, matrix_bytes,
                                          wwr::wwrMemcpyDeviceToDevice, stream));
  };

  // Step 1.
  FBF_TRY(riemannian_gradient(grad));
  FBF_TRY(reset());
  R gg = FBF_TRY(half_norm_sq(grad));

  for (int32_t iteration = 0;; ++iteration) {
    info.iterations = iteration;
    info.gradient_norm_sq = static_cast<double>(gg);
    FBF_LOG_DEBUG("cg_unitary: iteration {}, <G, G> {:.3e}, last step {:.3e}", iteration,
                  info.gradient_norm_sq, info.last_step);

    // Step 3, checked after every step, the last one included.
    if (gg < options.gradient_tolerance) {
      return info;
    }
    if (iteration == options.max_iterations) {
      return err_max_iterations_exceeded("cg_unitary: not converged in {} iterations (<G, G> = "
                                         "{:.3e}, tolerance {:.3e})",
                                         options.max_iterations, info.gradient_norm_sq,
                                         static_cast<double>(options.gradient_tolerance));
    }

    // Step 2's periodic reset; step 0 has just made one.
    if (iteration > 0 && iteration % reset_period == 0) {
      FBF_TRY(reset());
    }

    // Step 4. With no extremum along a conjugate direction, the gradient's is
    // still worth trying: the standard recovery, and Remark 2's reason too.
    R mu = FBF_TRY(geodesic_search<T>(res, n, w, cost, direction, options, s));
    if (!(mu > R{0}) && !steepest) {
      ++info.line_search_failures;
      FBF_TRY(reset());
      mu = FBF_TRY(geodesic_search<T>(res, n, w, cost, direction, options, s));
    }
    if (!(mu > R{0})) {
      return err_divergence("cg_unitary: no step along the gradient at iteration {} (<G, G> = "
                            "{:.3e}): the search found no extremum",
                            iteration, info.gradient_norm_sq);
    }
    info.last_step = static_cast<double>(mu);

    // Step 5: W <- exp(sign mu H) W, from the decomposition the search left.
    FBF_TRY(skew_expm<T>(res, n, s.eigvec.data(), s.eigval.data(), sign * mu, s.rotation.data()));
    FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n,
                                             &kOne<T>, s.rotation.data(), n, w, n, &kZero<T>,
                                             s.trial.data(), n)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(w, s.trial.data(), matrix_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));

    // Step 6: G' and gamma = <G' - G, G'> / <G, G>, eq. (10). Both inner
    // products carry the same 0.5, so the raw trace over 2 <G, G> is gamma.
    FBF_TRY(riemannian_gradient(grad_next));
    const R gg_next = FBF_TRY(half_norm_sq(grad_next));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s.tmp.data(), grad_next, matrix_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::axpy<T, int64_t>(blas, nn, &kNegOne<T>, grad, 1, s.tmp.data(), 1)));
    const R numerator = FBF_TRY(frobenius_dot<T>(blas, nn, s.tmp.data(), grad_next));
    const T gamma = cg_scalar<T>(gg > R{0} ? numerator / (kTwo<R> * gg) : R{0});

    // H <- G' + gamma H.
    FBF_TRY(gpu_status(wwr::scal<T, int64_t>(blas, nn, &gamma, s.dir.data(), 1)));
    FBF_TRY(gpu_status(wwr::axpy<T, int64_t>(blas, nn, &kOne<T>, grad_next, 1, s.dir.data(), 1)));
    steepest = false;
    std::swap(grad, grad_next);
    gg = gg_next;

    // Step 7, Remark 2: G is the ascent direction, so <H, G> > 0 is what makes
    // the step along sign * H go down (up) at all.
    const R hg = FBF_TRY(frobenius_dot<T>(blas, nn, s.dir.data(), grad));
    if (!(hg > R{0})) {
      FBF_TRY(reset());
    }
  }
}

} // namespace free_breakfast
