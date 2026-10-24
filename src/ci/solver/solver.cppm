/// @file
/// @brief The lowest roots of the CI Hamiltonian of one spin, through
/// davidson_solve over any sigma that is a LinearOperator, and S² on a CI
/// vector.
///
/// The README has the spin control: a penalty λ (S² − S(S+1)) on H, S taken
/// from the sector, and why that over filtering the roots afterwards.

export module free_breakfast.ci.solver;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync
import free_breakfast.core.resources;
// Re-exported: CiSolverOptions and CiRoots hold its options and info, and a
// sigma is written against its LinearOperator.
export import free_breakfast.recipes.davidson;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast {

// ci_solve's preconditioner for one column on the device: correction_i =
// residual_i / (theta − shifted_i), held off zero at ±1e-3, queued on
// res.stream(). Not exported: only ci_solve calls it, but its body is in the
// implementation unit, the one place the bridge header may go.
[[nodiscard]] Status ci_precondition(DeviceResources &res, int32_t n, double theta,
                                     const double *shifted, const double *residual,
                                     double *correction);

// ci_solve's @p count orthonormal guess columns into the n × count device
// block @p guess: unit vectors on the determinants of lowest @p diagonal
// (device, n), ties to the lower address, each with a 1e-3 deterministic
// spread over every determinant, then modified Gram-Schmidt twice. Queued on
// res.stream() with res.blas(), whose pointer mode it holds at host.
[[nodiscard]] Status ci_guesses(DeviceResources &res, int32_t n, int32_t count,
                                const double *diagonal, double *guess);

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

  /// @brief Y = S² X for the size() × @p k column-major device blocks @p x and
  /// @p y, leading dimension size(), which must not overlap. Queued on
  /// res.stream().
  ///
  /// @return InvalidConfig for another device, or a negative @p k or
  ///         k · size() past an int32_t; or the launch's GpuRuntime error.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief Y += λ (S² − @p target) X: the spin penalty, added into a sigma's
  /// output. As apply otherwise.
  [[nodiscard]] Status penalize(DeviceResources &res, int32_t k, double lambda, double target,
                                const double *x, double *y);

  /// @brief shifted = diagonal + λ (diag(S²) − @p target) for the size()
  /// device doubles @p diagonal and @p shifted (which may be the same): the
  /// diagonal of a penalized H, ci_solve's preconditioner's. diag(S²) is
  /// S_z (S_z + 1) + N_β less the orbitals doubly occupied. Queued on
  /// res.stream().
  ///
  /// @return InvalidConfig for another device; or the launch's GpuRuntime
  ///         error.
  [[nodiscard]] Status penalize_diagonal(DeviceResources &res, double lambda, double target,
                                         const double *diagonal, double *shifted);

  /// @brief The number of determinants, alpha.size() · beta.size().
  [[nodiscard]] int32_t size() const noexcept { return n_alpha_ * n_beta_; }

private:
  CiSpinSquare(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
               int32_t alpha_per, double constant);

  Status launch(DeviceResources &res, int32_t k, double shift, double scale, bool accumulate,
                const double *x, double *y);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  double constant_; // S_z (S_z + 1) + N_β
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
  /// @brief What davidson_solve did.
  DavidsonInfo davidson;
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
      static_cast<size_t>(n_roots) > dim || !(options.spin_penalty >= 0.0)) {
    return err_config("ci_solve: needs one orbital count, at most an int32_t of determinants, "
                      "n_roots in [1, {}] and spin_penalty >= 0 (orbitals {} and {}, n_roots {}, "
                      "spin_penalty {})",
                      dim, alpha.n_orbitals(), beta.n_orbitals(), n_roots, options.spin_penalty);
  }
  const auto n = static_cast<int32_t>(dim);
  const auto roots = static_cast<size_t>(n_roots);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const double penalty = options.spin_penalty;
  // S(S+1) with 2S = |N_α − N_β|.
  const int32_t two_s = std::abs(alpha.n_electrons() - beta.n_electrons());
  const double target = 0.25 * two_s * (two_s + 2);

  // Guesses on the 2 · n_roots determinants of lowest diag(H): see ci_guesses
  // and the README for the spread. The subspace holds up to 16 · n_roots.
  const auto guess_count = static_cast<int32_t>(std::min(dim, 2 * roots));
  const int32_t max_subspace =
      std::max(2 * n_roots, static_cast<int32_t>(std::min(dim, 16 * roots)));
  Buffers::device<double> guess(dim * static_cast<size_t>(guess_count), res.device());
  Buffers::device<double> shifted(dim, res.device());
  Buffers::device<double> values(roots, res.device());
  Buffers::device<double> vectors(dim * roots, res.device());
  FBF_TRY(ci_guesses(res, n, guess_count, diagonal, guess.data()));

  // H + λ (S² − S(S+1)) through the caller's sigma, the penalty added into
  // its output on the device, and diag(H + penalty) for the preconditioner.
  CiSpinSquare spin = FBF_TRY(CiSpinSquare::create(res, alpha, beta));
  FBF_TRY(spin.penalize_diagonal(res, penalty, target, diagonal, shifted.data()));
  struct Penalized {
    Op &sigma;
    CiSpinSquare &spin;
    double penalty;
    double target;

    Status apply(DeviceResources &r, int32_t k, const double *x, double *y) {
      FBF_TRY(sigma.apply(r, k, x, y));
      if (penalty == 0.0) {
        return {};
      }
      return spin.penalize(r, k, penalty, target, x, y);
    }
  };
  Penalized op{sigma, spin, penalty, target};

  // r / (θ − diag(H + penalty)) on the device, one launch per column, the
  // denominator kept off zero as the test's dense preconditioner is: a root
  // concentrated on one determinant has θ ≈ its diagonal.
  const auto precondition = [&](DeviceResources &r, std::span<const double> theta,
                                const double *residual, double *correction) -> Status {
    for (size_t col = 0; col < theta.size(); ++col) {
      FBF_TRY(ci_precondition(r, n, theta[col], shifted.data(), residual + col * dim,
                              correction + col * dim));
    }
    return {};
  };

  const auto info =
      davidson_solve<double>(res, op, precondition, n, n_roots, max_subspace, guess.data(),
                             guess_count, values.data(), vectors.data(), options.davidson);
  if (!info) {
    return std::unexpected{info.error()};
  }

  // ⟨S²⟩ per root on the device, S² X into the guesses' block (davidson_solve
  // has copied them, and guess_count >= n_roots), and θ less the penalty's
  // ⟨λ (S² − S(S+1))⟩: θ is the unit Ritz vector's Rayleigh quotient of the
  // penalized H, so this is its quotient of H itself.
  std::vector<double> theta(roots);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta.data(), values.data(), roots * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(spin.apply(res, n_roots, vectors.data(), guess.data()));
  CiRoots out{std::vector<double>(roots), std::vector<double>(roots), std::move(vectors), *info};
  {
    const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                            CreatePolicy<wwr::wwrblasStatus_t>>
        host_scalars{res.blas().get(), wwr::WWRBLAS_POINTER_MODE_HOST};
    for (size_t r = 0; r < roots; ++r) {
      FBF_TRY(
          gpu_status(wwr::dot<double, int32_t>(res.blas().get(), n, out.vectors.data() + r * dim, 1,
                                               guess.data() + r * dim, 1, &out.spin_squares[r])));
    }
  }
  FBF_TRY(res.sync());
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
