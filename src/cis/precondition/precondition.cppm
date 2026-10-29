/// @file
/// @brief CIS's Davidson start and preconditioner over the orbital-energy
/// diagonal Δε_ia = ε_a − ε_i: the diagonal itself, the unit-vector guess on
/// its lowest entries, and the correction r / (θ − Δε).
///
/// An element ia of an n_ov vector is i · n_virt + a: row-major (i, a). The
/// README has why Δε and not diag(A), and the clamp.

export module free_breakfast.cis.precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;

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

private:
  int32_t dev_idx_;
  int32_t n_ov_;
  const double *delta_;
  // The concept calls the preconditioner const; θ is per-call scratch it
  // overwrites, allocated once so no call allocates.
  mutable Buffers::device<double> theta_;
};

} // namespace free_breakfast::cis
