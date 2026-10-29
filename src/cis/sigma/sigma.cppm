/// @file
/// @brief The integral-direct CIS operator σ = A b over the n_ov excitation
/// vectors of an RHF reference, as a LinearOperator, and the method's small
/// vocabulary: the spin (cis::Spin) and the knobs of a solve (cis::Options).
///
/// The README has the five steps of a column, why they are gemms and not
/// gemmwich, and the screen.

export module free_breakfast.cis.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.davidson; // DavidsonOptions
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

export namespace free_breakfast::cis {

/// @brief Which excitations of the closed-shell reference: the singlets,
/// A = Δε + 2 (ia|jb) − (ij|ab), or the triplets (Ms = 0 components),
/// A = Δε − (ij|ab).
enum class Spin : int32_t {
  Singlet, ///< The singlet excitations.
  Triplet, ///< The triplet excitations.
};

/// @brief The knobs of one CIS solve.
struct Options {
  /// @brief The roots wanted, the lowest of A, at least 1.
  int32_t n_states = 5;
  /// @brief Singlets or triplets.
  Spin spin = Spin::Singlet;
  /// @brief Passed to davidson_solve. Its residual_tolerance is relative,
  /// ||A x − θ x|| <= tol · max(|θ|, ||H||), and an excitation energy and the
  /// subspace's largest Ritz value are both of order one hartree, so 1e-6 is a
  /// residual of ~1e-6 hartree: an energy good to its square over the gap,
  /// ~1e-11, and a vector (so an oscillator strength) to ~1e-6 over the gap.
  DavidsonOptions davidson{.residual_tolerance = 1e-6};
};

/// @brief σ = A b on the device, integral-direct: a
/// `LinearOperator<Sigma, double>` over blocks of n_ov = n_occ · n_virt
/// excitation vectors, element ia at i · n_virt + a.
///
/// Per column, with C_occ and C_virt the occupied and virtual columns of the
/// MO coefficients and w = 2, R = G for a singlet, w = −1, R = K for a
/// triplet:
///
/// 1. D = C_occ X C_virtᵀ, the spherical transition density (two gemms);
/// 2. its Cartesian form, U D Uᵀ (MatrixAOPair::spherical_to_cartesian);
/// 3. G = J[D] − K[D]/2 and K[D] (nonsymmetric_jk), at its default screens;
/// 4. R's spherical form, Uᵀ R U (MatrixAOPair::cartesian_to_spherical);
/// 5. σ = Δε ∘ b + w · C_occᵀ R C_virt (a dgmm, then two gemms, the second
///    accumulating into σ).
///
/// Columns run one after another: J's and K's classes hold one build's scratch.
/// Every buffer apply writes is allocated by create, which also grows
/// res.device_workspace() to the one block nonsymmetric_jk takes, so apply
/// allocates nothing.
///
/// Borrows the MO coefficients, the Δε vector and the J and K classes: the
/// caller keeps them alive and unchanged while the sigma is in use, and runs
/// nothing else over the classes during an apply. Move-only; the buffers are
/// on the device of the DeviceResources create was given, which every apply
/// must pass again.
class Sigma {
public:
  /// @brief A sigma over the RHF orbitals @p c (square, AO by MO, spherical,
  /// the first @p n_occ columns occupied), with @p delta the device Δε of
  /// cis::diagonal over the same split, for the excitations of @p spin.
  ///
  /// @p u is @p basis's Cartesian-to-spherical transform
  /// (upload_spherical_transform_matrix; pipeline.spherical_transform's
  /// SphericalTransformOutput::u), @p j_classes and @p k_classes J's and K's
  /// classes of @p basis (pipeline.eri's IntegralDirect), as nonsymmetric_jk
  /// takes them. A failed allocation aborts (CreatePolicy).
  ///
  /// @return InvalidConfig when @p c is on another device or is not
  ///         nao_sphe × nao_sphe, when @p delta is null, or unless
  ///         1 <= n_occ < nao_sphe with n_occ · (nao_sphe − n_occ) fitting an
  ///         int32_t.
  [[nodiscard]] static Result<Sigma> create(DeviceResources &res, const BasisSet &basis,
                                            std::shared_ptr<const Buffers::device<double>> u,
                                            std::span<CoulombPairClass> j_classes,
                                            std::span<OrderedPairClass> k_classes,
                                            const SphericalMatrixAO &c, int32_t n_occ,
                                            const double *delta, Spin spin);

  Sigma(Sigma &&) noexcept = default;
  Sigma &operator=(Sigma &&) noexcept = default;
  Sigma(const Sigma &) = delete;
  Sigma &operator=(const Sigma &) = delete;
  ~Sigma() = default;

  /// @brief Y = A X for the n_ov × @p k column-major device blocks @p x and
  /// @p y, leading dimension n_ov, one column at a time. Queued on
  /// res.stream() with res.blas(), whose pointer mode must be host (its
  /// default, and what davidson_solve holds); no host synchronization.
  ///
  /// @return InvalidConfig for another device or a negative @p k; else
  ///         nonsymmetric_jk's refusals, or the GpuRuntime error of the first
  ///         BLAS call or build that failed, which leaves @p y unspecified.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief n_ov = n_occ · n_virt, the operator's order.
  [[nodiscard]] int32_t size() const noexcept { return n_occ_ * n_virt_; }

private:
  Sigma(DeviceResources &res, const BasisSet &basis,
        std::shared_ptr<const Buffers::device<double>> u, std::span<CoulombPairClass> j_classes,
        std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c, int32_t n_occ,
        const double *delta, Spin spin);

  int32_t dev_idx_;
  int32_t nao_; // spherical, = MOs
  int32_t n_occ_;
  int32_t n_virt_;
  Spin spin_;
  const double *c_;
  const double *delta_;
  std::span<CoulombPairClass> j_classes_;
  std::span<OrderedPairClass> k_classes_;
  // Held across nonsymmetric_jk, which takes res.device_workspace(), so they
  // are the sigma's own (docs/style.md "Scratch memory"): D in both layouts,
  // R in both, the other of G and K, and the nao × n_occ half-transform
  // shared by steps 1 and 5.
  MatrixAOPair density_;
  MatrixAOPair response_;
  CartesianMatrixAO other_;
  Buffers::device<double> half_;
};

} // namespace free_breakfast::cis
