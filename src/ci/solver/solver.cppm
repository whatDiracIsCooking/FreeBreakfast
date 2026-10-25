/// @file
/// @brief The lowest roots of the CI Hamiltonian of one spin, through
/// davidson_solve over any sigma that is a LinearOperator, and S² on a CI
/// vector.
///
/// The README has the spin control: a penalty λ (S² − S(S+1)) on H, S taken
/// from the sector, and why that over filtering the roots afterwards; and why
/// a singlet solve runs on the symmetric layout's packed vectors.

export module free_breakfast.ci.solver;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync
import free_breakfast.core.resources;
// Re-exported: CiSolverOptions and CiRoots hold its options and info, and a
// sigma is written against its LinearOperator.
export import free_breakfast.recipes.davidson;
// orthonormalize, for stage 2's start; its target comes PUBLIC through
// recipes::davidson's.
import free_breakfast.recipes.orthogonalize;
import free_breakfast.ci.strings;
// CiLayout and ci_unpack, for ci_solve's symmetric path.
import free_breakfast.ci.sigma;

#include "diagnostics.h"

namespace free_breakfast {

// ci_solve's preconditioner on the device, per column of the theta.size()
// columns of the n-long blocks @p ritz_vectors (X), @p residual and
// @p correction: δ = M⁻¹ r, M = θ − shifted held off zero at ±1e-3, or with
// @p olsen Olsen's δ = M⁻¹ (r − ε x), ε = x^T M⁻¹ r / x^T M⁻¹ x. Queued on
// res.stream(); Olsen's dots run in device pointer mode into
// res.device_workspace(), so nothing comes back to the host. Not exported:
// only ci_solve calls it, but its body is in the implementation unit, the one
// place the bridge header may go.
[[nodiscard]] Status ci_precondition(DeviceResources &res, int32_t n, std::span<const double> theta,
                                     const double *shifted, const double *ritz_vectors,
                                     const double *residual, double *correction, bool olsen);

// ci_solve's @p count orthonormal guess columns into the n × count device
// block @p guess: unit vectors on the determinants of lowest @p diagonal
// (device, n), ties to the lower address, each with a 1e-3 deterministic
// spread over every determinant, then modified Gram-Schmidt twice, or with
// @p canonical orthonormalize (Numerical if a column drops, which
// independent guesses never should). Queued on res.stream() with
// res.blas(), whose pointer mode it holds at host.
[[nodiscard]] Status ci_guesses(DeviceResources &res, int32_t n, int32_t count,
                                const double *diagonal, double *guess, bool canonical);

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief S² c on the host, for a CI vector over two StringSets in
/// host_sigma's layout.
///
/// S² = S_z (S_z + 1) + S₋S₊, and S₋S₊ = N_β − Σ_pq E^β_pq E^α_qp, so both
/// terms go through the sets' own replacement lists: nothing leaves the
/// (N_α, N_β) sector. Cost is O(dim · N_α (n − N_α + 1)).
///
/// @param c The CI vector, alpha.size() · beta.size().
/// @param s2c Written with S² c, the same size; must not overlap @p c.
/// @return InvalidConfig when the sets span different orbital counts or a span
///         has the wrong size; nothing is written then.
[[nodiscard]] Status host_spin_square(const StringSet &alpha, const StringSet &beta,
                                      std::span<const double> c, std::span<double> s2c);

/// @brief S² on the device, for host_spin_square's operator and layout: a
/// `LinearOperator<CiSpinSquare, double>` over blocks of CI vectors, and the
/// spin penalty ci_solve adds into its sigma's output.
///
/// create uploads the α set's replacement lists and, per β string, a table of
/// its replacements by (p, q), so each α link a†_p a_q finds its β partner
/// a†_q a_p in one read. Both calls gather one thread per row over the row's
/// own links, since S² is real symmetric: nothing is atomic and the result
/// does not depend on the run. Cost is O(k · dim · N_α (n − N_α + 1)).
///
/// Over one StringSet for both spins, apply, penalize and penalize_diagonal
/// also take CiLayout::symmetric (see ci.sigma): S² commutes with the swap of
/// the two strings, so it maps the packed vectors to packed vectors.
///
/// Move-only; the buffers are on the device of the DeviceResources create was
/// given, which every call must pass again.
class CiSpinSquare {
public:
  /// @brief Upload @p alpha's lists and @p beta's slot tables to @p res's
  /// device. A failed allocation aborts (CreatePolicy).
  ///
  /// @return InvalidConfig when the two sets span different orbital counts or
  ///         alpha.size() · beta.size() does not fit an int32_t; or the
  ///         GpuRuntime error of an upload.
  [[nodiscard]] static Result<CiSpinSquare> create(DeviceResources &res, const StringSet &alpha,
                                                   const StringSet &beta);

  CiSpinSquare(CiSpinSquare &&) noexcept = default;
  CiSpinSquare &operator=(CiSpinSquare &&) noexcept = default;
  CiSpinSquare(const CiSpinSquare &) = delete;
  CiSpinSquare &operator=(const CiSpinSquare &) = delete;
  ~CiSpinSquare() = default;

  /// @brief Y = S² X for the size(@p layout) × @p k column-major device blocks
  /// @p x and @p y, leading dimension size(@p layout), which must not
  /// overlap. Queued on res.stream().
  ///
  /// @return InvalidConfig for another device, a negative @p k or
  ///         k · size(@p layout) past an int32_t, or CiLayout::symmetric over
  ///         two different sets; or the launch's GpuRuntime error.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y,
                             CiLayout layout = CiLayout::full);

  /// @brief Y += λ (S² − @p target) X: the spin penalty, added into a sigma's
  /// output. As apply otherwise.
  [[nodiscard]] Status penalize(DeviceResources &res, int32_t k, double lambda, double target,
                                const double *x, double *y, CiLayout layout = CiLayout::full);

  /// @brief shifted = diagonal + λ (diag(S²) − @p target), for the size()
  /// device doubles @p diagonal, into the size(@p layout) device doubles
  /// @p shifted: the diagonal of a penalized H, ci_solve's preconditioner's.
  /// In CiLayout::symmetric, shifted holds the triangle's determinants,
  /// unscaled. diag(S²) is S_z (S_z + 1) + N_β less the orbitals doubly
  /// occupied. @p shifted may be @p diagonal in the full layout only. Queued
  /// on res.stream().
  ///
  /// @return InvalidConfig for another device, or CiLayout::symmetric over two
  ///         different sets; or the launch's GpuRuntime error.
  [[nodiscard]] Status penalize_diagonal(DeviceResources &res, double lambda, double target,
                                         const double *diagonal, double *shifted,
                                         CiLayout layout = CiLayout::full);

  /// @brief The number of determinants, alpha.size() · beta.size(), or for
  /// CiLayout::symmetric the triangle's, alpha.size() (alpha.size() + 1) / 2.
  [[nodiscard]] int32_t size(CiLayout layout = CiLayout::full) const noexcept {
    return layout == CiLayout::symmetric ? n_alpha_ * (n_alpha_ + 1) / 2 : n_alpha_ * n_beta_;
  }

private:
  CiSpinSquare(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
               int32_t alpha_per, double constant, bool one_set);

  Status check(DeviceResources &res, CiLayout layout) const;
  Status launch(DeviceResources &res, int32_t k, double shift, double scale, bool accumulate,
                const double *x, double *y, CiLayout layout);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  double constant_; // S_z (S_z + 1) + N_β
  bool one_set_;    // alpha and beta are the same (n, N): CiLayout::symmetric applies
  // Three int32_t per α replacement: target, its β partner's slot, sign.
  Buffers::device<int32_t> alpha_links_;
  // n_beta × n × n: sign · (target + 1) of a†_p a_q on each β string, or 0.
  Buffers::device<int32_t> beta_slots_;
};

/// @brief The knobs of one ci_solve call.
struct CiSolverOptions {
  /// @brief λ in H + λ (S² − S(S+1)), hartree; S = |N_α − N_β| / 2. A state of
  /// spin S' > S rises by λ (S'(S'+1) − S(S+1)) ≥ 2 λ (S+1), so it must exceed
  /// the span of the wanted roots over the lowest unwanted one. 0 keeps every
  /// spin.
  double spin_penalty = 1.0;
  /// @brief Passed to davidson_solve. The residual tolerance is tighter than
  /// its default because a vector's error is first order in the residual, and
  /// the 1-RDM is read from the vector: 1e-12 relative is ~1e-10 hartree at
  /// molecular energies.
  DavidsonOptions davidson{.residual_tolerance = 1e-12};
  /// @brief The relative residual a first, single-precision stage runs to,
  /// when the sigma has an apply_fp32 (CiSigma does): davidson_solve over
  /// that, then again over apply from its Ritz vectors, to
  /// davidson.residual_tolerance. 0 runs apply alone, as does any value at or
  /// below that tolerance. FP32 sigma's noise stalls Davidson near 1e-7, and
  /// a stage that stalls spends its max_iterations; the README has the
  /// measurement behind the default.
  double fp32_tolerance = 1e-6;
  /// @brief Olsen's correction M⁻¹ (r − ε x), ε = x^T M⁻¹ r / x^T M⁻¹ x, in
  /// place of the plain M⁻¹ r, M = θ − diag(H + penalty). Off by default:
  /// over the diagonal M it is slower than the plain correction on most of
  /// the README's bench solves.
  bool olsen_correction = false;
  /// @brief Start the FP64 stage from the FP32 stage's Ritz vectors X and
  /// the iteration's before them, X_prev, orthonormalized together (nearly
  /// dependent columns dropped), in place of X alone. Off by default: over
  /// the README's N2 bench it was slower than X alone on most two-stage
  /// solves.
  bool restart_with_previous = false;
};

/// @brief The roots a ci_solve converged, ascending in energy.
struct CiRoots {
  /// @brief ⟨H⟩ per root, E_core included: the Ritz value less its penalty.
  std::vector<double> energies;
  /// @brief ⟨S²⟩ per root.
  std::vector<double> spin_squares;
  /// @brief The unit CI vectors on the device of the ci_solve call, root r at
  /// [r · dim, (r + 1) · dim), each in host_sigma's layout.
  Buffers::device<double> vectors;
  /// @brief What the final davidson_solve did, over the sigma's apply.
  DavidsonInfo davidson;
  /// @brief Iterations of the single-precision stage before it, 0 when there
  /// was none (see CiSolverOptions::fp32_tolerance).
  int32_t fp32_iterations = 0;
};

/// @brief The @p n_roots lowest roots of spin S = |N_α − N_β| / 2 of the CI
/// Hamiltonian that @p sigma applies.
///
/// davidson_solve runs on H + λ (S² − S(S+1)) (λ = options.spin_penalty),
/// which leaves spin S where it is and lifts every higher spin, from guesses
/// on the 2 · n_roots determinants of lowest diag(H) (each with a small spread
/// over every determinant, so no spatial symmetry is missed), preconditioned by
/// r / (θ − diag(H + penalty)). S is the sector's lowest spin, so choosing the
/// spin is choosing N_α − N_β = 2S. Everything runs on the device: the
/// guesses and the penalized diagonal are built there from @p diagonal, the
/// penalty (a CiSpinSquare) and the preconditioner run there per iteration,
/// ⟨S²⟩ is taken there, and the roots stay there. Only the n_roots Ritz values
/// and ⟨S²⟩s come to the host.
///
/// A singlet solve, over one StringSet for both spins with a penalty, by a
/// sigma that takes a CiLayout (CiSigma), runs on CiLayout::symmetric's packed
/// vectors: every vector of even S is symmetric under the swap of the two
/// strings, so the Davidson keeps to that subspace exactly, its vectors, A V
/// and sigma scratch are about half as long, and quintets are the nearest
/// spin the penalty has to lift. The roots are unpacked at the end, so
/// CiRoots is the same either way.
///
/// A sigma that also has `apply_fp32(res, k, x, y)` (CiSigma) is solved in two
/// stages: davidson_solve over apply_fp32 to options.fp32_tolerance, then over
/// apply, restarted from the first stage's Ritz vectors, to
/// options.davidson's tolerance. The second stage is the whole solve as it
/// would be without the first, from better guesses, so the roots are as
/// accurate as an FP64-only solve's.
///
/// @param sigma    Applies H to a device block of CI vectors (host_sigma's
///                 layout); a LinearOperator over dim = alpha.size() ·
///                 beta.size().
/// @param alpha, beta The sector's strings.
/// @param diagonal diag(H) on the device, dim doubles, as CiSigma::diagonal
///                 (or host_diagonal) writes it.
/// @param n_roots  Roots wanted, in [1, dim].
/// @return The roots. Otherwise InvalidConfig for a shape or option out of
///         range; davidson_solve's error (MaxIterationsExceeded, Divergence,
///         or @p sigma's own); or Numerical when, with a penalty, a root's ⟨S²⟩
///         is more than 1e-6 off S(S+1): a state of another spin lay too far
///         below for the penalty to lift.
template <LinearOperator<double> Op>
[[nodiscard]] Result<CiRoots> ci_solve(DeviceResources &res, Op &sigma, const StringSet &alpha,
                                       const StringSet &beta, const double *const diagonal,
                                       const int32_t n_roots, const CiSolverOptions &options = {}) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (alpha.n_orbitals() != beta.n_orbitals() ||
      dim > static_cast<size_t>(std::numeric_limits<int32_t>::max()) || n_roots < 1 ||
      static_cast<size_t>(n_roots) > dim || !(options.spin_penalty >= 0.0) ||
      !(options.fp32_tolerance >= 0.0)) {
    return err_config("ci_solve: needs one orbital count, at most an int32_t of determinants, "
                      "n_roots in [1, {}], spin_penalty >= 0 and fp32_tolerance >= 0 (orbitals {} "
                      "and {}, n_roots {}, spin_penalty {}, fp32_tolerance {})",
                      dim, alpha.n_orbitals(), beta.n_orbitals(), n_roots, options.spin_penalty,
                      options.fp32_tolerance);
  }
  const auto roots = static_cast<size_t>(n_roots);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const double penalty = options.spin_penalty;
  // S(S+1) with 2S = |N_α − N_β|.
  const int32_t two_s = std::abs(alpha.n_electrons() - beta.n_electrons());
  const double target = 0.25 * two_s * (two_s + 2);

  // A singlet solve by a sigma that takes a layout runs on the symmetric
  // layout's packed vectors (see the README); with no penalty the caller
  // wants every spin, odd S included, so the full layout. n is the length the
  // Davidson sees.
  constexpr bool takes_layout = requires(const double *x, double *y) {
    sigma.apply(res, n_roots, x, y, CiLayout::symmetric);
  };
  const size_t packed_dim = static_cast<size_t>(alpha.size()) * (alpha.size() + 1) / 2;
  const CiLayout layout = takes_layout && two_s == 0 && alpha.n_electrons() == beta.n_electrons() &&
                                  penalty > 0.0 && roots <= packed_dim
                              ? CiLayout::symmetric
                              : CiLayout::full;
  const size_t length = layout == CiLayout::symmetric ? packed_dim : dim;
  const auto n = static_cast<int32_t>(length);

  // Guesses on the 2 · n_roots determinants of lowest diag(H): see ci_guesses
  // and the README for the spread. In the symmetric layout they are triangle
  // elements, each a determinant and its mirror. The subspace holds up to
  // 16 · n_roots.
  const auto guess_count = static_cast<int32_t>(std::min(length, 2 * roots));
  const int32_t max_subspace =
      std::max(2 * n_roots, static_cast<int32_t>(std::min(length, 16 * roots)));
  Buffers::device<double> guess(length * static_cast<size_t>(guess_count), res.device());
  Buffers::device<double> shifted(length, res.device());
  Buffers::device<double> values(roots, res.device());
  Buffers::device<double> vectors(length * roots, res.device());
  CiSpinSquare spin = FBF_TRY(CiSpinSquare::create(res, alpha, beta));
  if (layout == CiLayout::symmetric) {
    // diag(H) over the triangle, through penalize_diagonal with no penalty,
    // into shifted until the penalized one overwrites it.
    FBF_TRY(spin.penalize_diagonal(res, 0.0, target, diagonal, shifted.data(), layout));
    FBF_TRY(ci_guesses(res, n, guess_count, shifted.data(), guess.data(),
                       options.davidson.canonical_orthonormalization));
  } else {
    FBF_TRY(ci_guesses(res, n, guess_count, diagonal, guess.data(),
                       options.davidson.canonical_orthonormalization));
  }

  // H + λ (S² − S(S+1)) through the caller's sigma, the penalty added into
  // its output on the device, and diag(H + penalty) for the preconditioner.
  // fp32 picks the sigma's apply_fp32; applies counts the calls, one per
  // iteration, since every iteration adds a column.
  FBF_TRY(spin.penalize_diagonal(res, penalty, target, diagonal, shifted.data(), layout));
  struct Penalized {
    Op &sigma;
    CiSpinSquare &spin;
    double penalty;
    double target;
    CiLayout layout;
    bool fp32 = false;
    int32_t applies = 0;

    Status apply(DeviceResources &r, int32_t k, const double *x, double *y) {
      ++applies;
      if constexpr (requires { sigma.apply_fp32(r, k, x, y, layout); }) {
        FBF_TRY(fp32 ? sigma.apply_fp32(r, k, x, y, layout) : sigma.apply(r, k, x, y, layout));
      } else if constexpr (requires { sigma.apply(r, k, x, y, layout); }) {
        FBF_TRY(sigma.apply(r, k, x, y, layout));
      } else if constexpr (requires { sigma.apply_fp32(r, k, x, y); }) {
        FBF_TRY(fp32 ? sigma.apply_fp32(r, k, x, y) : sigma.apply(r, k, x, y));
      } else {
        FBF_TRY(sigma.apply(r, k, x, y));
      }
      if (penalty == 0.0) {
        return {};
      }
      return spin.penalize(r, k, penalty, target, x, y, layout);
    }
  };
  Penalized op{sigma, spin, penalty, target, layout};

  // M⁻¹ r over M = θ − diag(H + penalty) on the device, or Olsen's
  // M⁻¹ (r − ε x) with options.olsen_correction (see the README). The
  // denominator is kept off zero as the test's dense preconditioner is: a
  // root concentrated on one determinant has θ ≈ its diagonal.
  const auto precondition = [&](DeviceResources &r, std::span<const double> theta,
                                const double *ritz_vectors, const double *residual,
                                double *correction) -> Status {
    return ci_precondition(r, n, theta, shifted.data(), ritz_vectors, residual, correction,
                           options.olsen_correction);
  };

  // Stage 1, when the sigma has an apply_fp32: that, to the loose tolerance.
  // Running out of iterations or subspace there is no error, since stage 2
  // starts from whatever Ritz vectors it left. Stage 2 is the solve there
  // would be without stage 1, over apply to the caller's tolerance, only from
  // those vectors X (V S for an orthonormal V, so orthonormal) in place of
  // the guesses. With options.restart_with_previous, from [X, X_prev] instead,
  // X_prev the iteration's before X, orthonormalized, nearly dependent
  // columns dropped, so the restart keeps stage 1's last search direction.
  // X_prev is written into the guesses' second n_roots columns, which hold
  // 2 · n_roots whenever the space does.
  int32_t fp32_iterations = 0;
  int32_t start_count = guess_count;
  if constexpr (requires { sigma.apply_fp32(res, n_roots, guess.data(), vectors.data()); }) {
    if (options.fp32_tolerance > options.davidson.residual_tolerance) {
      DavidsonOptions loose = options.davidson;
      loose.residual_tolerance = options.fp32_tolerance;
      op.fp32 = true;
      const bool with_previous = options.restart_with_previous && guess_count >= 2 * n_roots;
      double *const previous = with_previous ? guess.data() + (length * roots) : nullptr;
      const auto stage =
          davidson_solve<double>(res, op, precondition, n, n_roots, max_subspace, guess.data(),
                                 guess_count, values.data(), vectors.data(), loose, previous);
      op.fp32 = false;
      if (!stage && stage.error().kind != ErrorKind::MaxIterationsExceeded &&
          stage.error().kind != ErrorKind::Divergence) {
        return std::unexpected{stage.error()};
      }
      fp32_iterations = op.applies;
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(guess.data(), vectors.data(),
                                             length * roots * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      start_count = n_roots;
      if (with_previous) {
        // X is orthonormal, so at least its n_roots columns survive.
        start_count = FBF_TRY(orthonormalize<double>(res, n, 2 * n_roots, guess.data(),
                                                     options.davidson.linear_dependence_floor));
        if (start_count < n_roots) {
          return err_numerical("ci_solve: orthonormalizing [X, X_prev] kept {} of {} columns, "
                               "fewer than the {} roots",
                               start_count, 2 * n_roots, n_roots);
        }
      }
    }
  }

  const auto info =
      davidson_solve<double>(res, op, precondition, n, n_roots, max_subspace, guess.data(),
                             start_count, values.data(), vectors.data(), options.davidson);
  if (!info) {
    return std::unexpected{info.error()};
  }

  // ⟨S²⟩ per root on the device, S² X into the guesses' block (davidson_solve
  // has copied them, and guess_count >= n_roots), and θ less the penalty's
  // ⟨λ (S² − S(S+1))⟩: θ is the unit Ritz vector's Rayleigh quotient of the
  // penalized H, so this is its quotient of H itself. The packing is an
  // isometry, so the dots are the same in either layout.
  std::vector<double> theta(roots);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta.data(), values.data(), roots * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(spin.apply(res, n_roots, vectors.data(), guess.data(), layout));
  std::vector<double> spin_squares(roots);
  {
    const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                            CreatePolicy<wwr::wwrblasStatus_t>>
        host_scalars{res.blas().get(), wwr::WWRBLAS_POINTER_MODE_HOST};
    for (size_t r = 0; r < roots; ++r) {
      FBF_TRY(
          gpu_status(wwr::dot<double, int32_t>(res.blas().get(), n, vectors.data() + r * length, 1,
                                               guess.data() + r * length, 1, &spin_squares[r])));
    }
  }
  // The roots in the full layout, as CiSigma::rdm1 and CiRdm::rdm12 take them.
  if (layout == CiLayout::symmetric) {
    Buffers::device<double> full(dim * roots, res.device());
    FBF_TRY(ci_unpack(res, alpha.size(), n_roots, vectors.data(), full.data()));
    vectors = std::move(full);
  }
  FBF_TRY(res.sync());
  CiRoots out{std::vector<double>(roots), std::move(spin_squares), std::move(vectors), *info,
              fp32_iterations};
  for (size_t r = 0; r < roots; ++r) {
    const double s2 = out.spin_squares[r];
    out.energies[r] = theta[r] - penalty * (s2 - target);
    if (penalty > 0.0 && std::abs(s2 - target) > 1e-6) {
      return err_numerical("ci_solve: root {} has <S^2> = {:.8f}, not S(S+1) = {}: a state of "
                           "another spin lies below the penalty's reach (spin_penalty {})",
                           r, s2, target, penalty);
    }
  }
  return out;
}

} // namespace free_breakfast
