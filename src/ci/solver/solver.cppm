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
  /// @brief The unit CI vectors, root r at [r · dim, (r + 1) · dim), each in
  /// host_sigma's layout.
  std::vector<double> vectors;
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
/// spin is choosing N_α − N_β = 2S. The penalty and the preconditioner run on
/// the host, a block copied each way per call.
///
/// @param sigma    Applies H to a device block of CI vectors (host_sigma's
///                 layout); a LinearOperator over dim = alpha.size() ·
///                 beta.size().
/// @param alpha, beta The sector's strings.
/// @param diagonal diag(H) on the host, dim long, as host_diagonal writes it.
/// @param n_roots  Roots wanted, in [1, dim].
/// @return The roots. Otherwise InvalidConfig for a shape or option out of
///         range; davidson_solve's error (MaxIterationsExceeded, Divergence,
///         or @p sigma's own); or Numerical when, with a penalty, a root's ⟨S²⟩
///         is more than 1e-6 off S(S+1): a state of another spin lay too far
///         below for the penalty to lift.
template <LinearOperator<double> Op>
[[nodiscard]] Result<CiRoots> ci_solve(DeviceResources &res, Op &sigma, const StringSet &alpha,
                                       const StringSet &beta,
                                       const std::span<const double> diagonal,
                                       const int32_t n_roots, const CiSolverOptions &options = {}) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (alpha.n_orbitals() != beta.n_orbitals() || diagonal.size() != dim ||
      dim > static_cast<size_t>(std::numeric_limits<int32_t>::max()) || n_roots < 1 ||
      static_cast<size_t>(n_roots) > dim || !(options.spin_penalty >= 0.0)) {
    return err_config("ci_solve: needs one orbital count, a diagonal of {} determinants, n_roots "
                      "in [1, {}] and spin_penalty >= 0 (orbitals {} and {}, diagonal {}, n_roots "
                      "{}, spin_penalty {})",
                      dim, dim, alpha.n_orbitals(), beta.n_orbitals(), diagonal.size(), n_roots,
                      options.spin_penalty);
  }
  const auto n = static_cast<int32_t>(dim);
  const auto roots = static_cast<size_t>(n_roots);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const double penalty = options.spin_penalty;
  // S(S+1) with 2S = |N_α − N_β|, and S_z (S_z + 1).
  const int32_t two_s = std::abs(alpha.n_electrons() - beta.n_electrons());
  const double target = 0.25 * two_s * (two_s + 2);
  const double two_sz = alpha.n_electrons() - beta.n_electrons();
  const double sz_term = 0.25 * two_sz * (two_sz + 2.0);

  // diag(H + penalty): diag(S²) is S_z (S_z + 1) + N_β less the doubly
  // occupied orbitals, the diagonal of Σ E^β_pp E^α_pp.
  std::vector<double> shifted(dim);
  for (size_t ia = 0; ia < static_cast<size_t>(alpha.size()); ++ia) {
    for (size_t ib = 0; ib < static_cast<size_t>(beta.size()); ++ib) {
      const size_t det = ia * static_cast<size_t>(beta.size()) + ib;
      const auto doubles = std::popcount(alpha.strings()[ia] & beta.strings()[ib]);
      shifted[det] = diagonal[det] + penalty * (sz_term + beta.n_electrons() - doubles - target);
    }
  }

  // Guesses: unit vectors on the determinants of lowest diag(H), ties to the
  // lower address, twice as many as the roots. A determinant has one spatial
  // symmetry, and a diagonal preconditioner and H both keep to it, so from
  // unit vectors alone a root of a symmetry the guesses miss never appears:
  // N₂'s second and third singlets are lost that way. So each guess also gets
  // a 1e-3 spread over every determinant, deterministic (a splitmix64 hash of
  // its position), and the columns are then orthonormalized.
  const auto guess_count = static_cast<int32_t>(std::min(dim, 2 * roots));
  const int32_t max_subspace =
      std::max(2 * n_roots, static_cast<int32_t>(std::min(dim, 16 * roots)));
  std::vector<size_t> order(dim);
  std::iota(order.begin(), order.end(), size_t{0});
  std::ranges::stable_sort(order, {}, [&](size_t i) { return diagonal[i]; });
  std::vector<double> guess_host(dim * static_cast<size_t>(guess_count));
  for (size_t i = 0; i < guess_host.size(); ++i) {
    uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;
    guess_host[i] = 1e-3 * (static_cast<double>(z >> 11U) * 0x1.0p-52 - 1.0); // [-1e-3, 1e-3)
  }
  for (size_t g = 0; g < static_cast<size_t>(guess_count); ++g) {
    const std::span<double> column{guess_host.data() + g * dim, dim};
    column[order[g]] += 1.0;
    // Modified Gram-Schmidt, twice, against the columns before.
    for (int32_t pass = 0; pass < 2; ++pass) {
      for (size_t h = 0; h < g; ++h) {
        const std::span<const double> earlier{guess_host.data() + h * dim, dim};
        const double overlap =
            std::inner_product(earlier.begin(), earlier.end(), column.begin(), 0.0);
        for (size_t i = 0; i < dim; ++i) {
          column[i] -= overlap * earlier[i];
        }
      }
    }
    const double norm =
        std::sqrt(std::inner_product(column.begin(), column.end(), column.begin(), 0.0));
    for (double &x : column) {
      x /= norm;
    }
  }
  Buffers::device<double> guess(guess_host.size(), res.device());
  Buffers::device<double> values(roots, res.device());
  Buffers::device<double> vectors(dim * roots, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(guess.data(), guess_host.data(),
                                         guess_host.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(res.sync());

  // H + λ (S² − S(S+1)) through the caller's sigma, the penalty on the host
  // in res.host_workspace(), which sigma has finished with by the sync.
  struct Penalized {
    Op &sigma;
    const StringSet &alpha;
    const StringSet &beta;
    double penalty;
    double target;
    size_t dim;

    Status apply(DeviceResources &r, int32_t k, const double *x, double *y) {
      FBF_TRY(sigma.apply(r, k, x, y));
      if (penalty == 0.0) {
        return {};
      }
      const size_t count = dim * static_cast<size_t>(k);
      const size_t bytes = count * sizeof(double);
      auto *xh = reinterpret_cast<double *>(r.host_workspace(3 * bytes).data());
      double *yh = xh + count;
      double *s2x = yh + count;
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(xh, x, bytes, wwr::wwrMemcpyDeviceToHost, r.stream())));
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(yh, y, bytes, wwr::wwrMemcpyDeviceToHost, r.stream())));
      FBF_TRY(r.sync());
      for (size_t col = 0; col < static_cast<size_t>(k); ++col) {
        const size_t at = col * dim;
        FBF_TRY(host_spin_square(alpha, beta, std::span<const double>{xh + at, dim},
                                 std::span<double>{s2x + at, dim}));
      }
      for (size_t i = 0; i < count; ++i) {
        yh[i] += penalty * (s2x[i] - target * xh[i]);
      }
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(y, yh, bytes, wwr::wwrMemcpyHostToDevice, r.stream())));
      return r.sync();
    }
  };
  Penalized op{sigma, alpha, beta, penalty, target, dim};

  // r / (θ − diag), kept off zero as the test's dense preconditioner is: a
  // root concentrated on one determinant has θ ≈ its diagonal.
  const auto precondition = [&](DeviceResources &r, std::span<const double> theta,
                                const double *residual, double *correction) -> Status {
    const size_t count = dim * theta.size();
    const size_t bytes = count * sizeof(double);
    auto *h = reinterpret_cast<double *>(r.host_workspace(bytes).data());
    FBF_TRY(gpu_status(
        wwr::wwrMemcpyAsync(h, residual, bytes, wwr::wwrMemcpyDeviceToHost, r.stream())));
    FBF_TRY(r.sync());
    for (size_t col = 0; col < theta.size(); ++col) {
      for (size_t i = 0; i < dim; ++i) {
        double denominator = theta[col] - shifted[i];
        if (std::abs(denominator) < 1e-3) {
          denominator = denominator < 0.0 ? -1e-3 : 1e-3;
        }
        h[col * dim + i] /= denominator;
      }
    }
    FBF_TRY(gpu_status(
        wwr::wwrMemcpyAsync(correction, h, bytes, wwr::wwrMemcpyHostToDevice, r.stream())));
    return r.sync();
  };

  const auto info =
      davidson_solve<double>(res, op, precondition, n, n_roots, max_subspace, guess.data(),
                             guess_count, values.data(), vectors.data(), options.davidson);
  if (!info) {
    return std::unexpected{info.error()};
  }

  CiRoots out{std::vector<double>(roots), std::vector<double>(roots),
              std::vector<double>(dim * roots), *info};
  std::vector<double> theta(roots);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta.data(), values.data(), roots * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.vectors.data(), vectors.data(),
                                         out.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  // ⟨S²⟩ per root, and θ less the penalty's ⟨λ (S² − S(S+1))⟩: θ is the
  // unit Ritz vector's Rayleigh quotient of the penalized H, so this is
  // its quotient of H itself.
  std::vector<double> s2c(dim);
  for (size_t r = 0; r < roots; ++r) {
    const std::span<const double> c{out.vectors.data() + r * dim, dim};
    FBF_TRY(host_spin_square(alpha, beta, c, s2c));
    const double s2 = std::inner_product(c.begin(), c.end(), s2c.begin(), 0.0);
    out.spin_squares[r] = s2;
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
