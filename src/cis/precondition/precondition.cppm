/// @file
/// @brief CIS's Davidson start and preconditioner over the orbital-energy
/// diagonal Δε_ia = ε_a − ε_i: the diagonal itself, the unit-vector guess on
/// its lowest entries, and the correction r / (θ − Δε), optionally with an
/// exact-block correction over clusters of near-degenerate Δε.
///
/// An element ia of an n_ov vector is i · n_virt + a: row-major (i, a). The
/// README has why Δε and not diag(A), the clamp, and the clusters.

export module free_breakfast.cis.precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.recipes.linear_operator; // LinearOperator

export namespace free_breakfast::cis {

/// @brief Δε_ia = ε_a − ε_i into @p delta (device, n_occ · n_virt), element
/// i · n_virt + a, from @p orbital_energies (device, n_occ + n_virt, the
/// occupied first). Queued on res.stream().
///
/// @return InvalidConfig unless n_occ >= 1, n_virt >= 1 and n_occ · n_virt
///         fits an int32_t.
[[nodiscard]] Status diagonal(DeviceResources &res, int32_t n_occ, int32_t n_virt,
                              const double *orbital_energies, double *delta);

/// @brief The @p guess_count lowest entries of the @p n_ov device @p delta as
/// unit vectors, into the column-major n_ov × guess_count device block
/// @p vectors: column g is 1 on the g-th lowest entry, ascending, ties to the
/// lower index, and 0 elsewhere. Orthonormal by construction, and
/// deterministic.
///
/// Reads @p delta back to the host for the select, so it synchronizes; it
/// runs once per solve.
///
/// @return InvalidConfig unless 1 <= guess_count <= n_ov.
[[nodiscard]] Status guess(DeviceResources &res, int32_t n_ov, const double *delta,
                           int32_t guess_count, double *vectors);

/// @brief The diagonal correction r / (θ − Δε) as a DavidsonPreconditioner:
/// each column c of the residual block divided element-wise by θ_c − Δε, the
/// denominator's magnitude held at or above 1e-3 with its sign kept.
///
/// With @ref cluster, the entries covered by a cluster of near-degenerate Δε
/// are corrected in that cluster's exact eigenbasis instead, with the same
/// clamp; the others keep r / (θ − Δε).
///
/// The Ritz values are uploaded into a device buffer of @p max_columns
/// allocated here, so a call allocates nothing. @p delta is borrowed, not
/// copied: the caller keeps it alive and unchanged while this is in use.
class Preconditioner {
public:
  /// @brief Over the @p n_ov device @p delta, for calls of at most
  /// @p max_columns columns (davidson_solve's n_roots), on res's device.
  Preconditioner(DeviceResources &res, int32_t n_ov, const double *delta, int32_t max_columns);

  /// @brief correction = residual / (θ − Δε), column by column, for the
  /// theta.size() n_ov-long columns of the device blocks; @p ritz_vectors is
  /// not read. Queued on res.stream(), then synchronized, since @p theta is
  /// pageable and its upload must land before the caller may change it.
  ///
  /// @return InvalidConfig for more than max_columns columns or another
  ///         device; a runtime failure, passed up.
  [[nodiscard]] Status operator()(DeviceResources &res, std::span<const double> theta,
                                  const double *ritz_vectors, const double *residual,
                                  double *correction) const;

  /// @brief Build the clustered exact-block correction, once, before the
  /// solve, replacing any built before.
  ///
  /// The @p scope lowest Δε, ascending with ties to the lower index, are
  /// grouped by single linkage: neighbours at most @p gap apart share a
  /// cluster, and a cluster of one is dropped. Each cluster's exact k × k
  /// block of A is read from @p sigma applied once to the unit vectors of
  /// every covered entry, symmetrized, and diagonalized
  /// (recipes.diagonalize). A call then projects a covered entry's residual
  /// onto its cluster's eigenvectors, divides by θ − λ under the same clamp,
  /// and projects back. @p gap = 0 builds nothing, and leaves every call
  /// exactly the diagonal correction.
  ///
  /// Grows res.device_workspace() to a call's scratch, so a call still
  /// allocates nothing. Synchronizes res.stream().
  ///
  /// @return InvalidConfig for another device, unless 1 <= scope <= n_ov, or
  ///         for a @p gap that is negative or NaN; else @p sigma's error, or
  ///         diagonalize's, or a runtime failure, any of which leaves no
  ///         cluster built.
  template <class Op>
    requires LinearOperator<Op, double>
  [[nodiscard]] Status cluster(DeviceResources &res, Op &sigma, int32_t scope, double gap) {
    // The unit vectors of the covered entries, n_ov × covered; empty when
    // nothing clusters.
    Result<Buffers::device<double>> units = group(res, scope, gap);
    if (!units) {
      return std::unexpected(std::move(units).error());
    }
    if (units->num_elements() == 0) {
      return {};
    }
    Buffers::device<double> applied(units->num_elements(), res.device());
    if (Status applied_ok = sigma.apply(res, covered_, units->data(), applied.data());
        !applied_ok) {
      covered_ = 0;
      return applied_ok;
    }
    return solve_blocks(res, applied.data());
  }

private:
  // Validates, drops any clusters built before, and groups the scope lowest
  // Δε: the covered entries' index_ and sizes_, and their unit vectors,
  // returned. Leaves covered_ at zero, and the block empty, when none
  // clusters.
  [[nodiscard]] Result<Buffers::device<double>> group(DeviceResources &res, int32_t scope,
                                                      double gap);
  // Each cluster's block of A from @p applied, the sigma of the unit vectors,
  // diagonalized into vectors_ and lambda_; covered_ set only on success.
  [[nodiscard]] Status solve_blocks(DeviceResources &res, const double *applied);

  int32_t dev_idx_;
  int32_t n_ov_;
  const double *delta_;
  // The concept calls the preconditioner const; θ is per-call scratch it
  // overwrites, allocated once so no call allocates.
  mutable Buffers::device<double> theta_;
  // The clusters, back to back: covered_ entries, cluster g the sizes_[g]
  // after those before it; their indices into the n_ov vector, their
  // eigenvalues, and their eigenvectors as the covered_ × covered_
  // block-diagonal column-major matrix, the zeros between blocks included so
  // a call is two gemms. covered_ is zero when there are none.
  int32_t covered_ = 0;
  std::vector<int32_t> sizes_;
  Buffers::device<int32_t> index_;
  Buffers::device<double> lambda_;
  Buffers::device<double> vectors_;
};

} // namespace free_breakfast::cis
