/// @file
/// @brief The CIS operator σ = A b over the n_ov excitation vectors of an RHF
/// reference, as a LinearOperator, integral-direct (cis::Sigma) or
/// density-fitted in the MO basis (cis::DfSigma), and the method's small
/// vocabulary: the spin (cis::Spin) and the knobs of a solve (cis::Options).
///
/// The README has the five steps of a direct column, why they are gemms and
/// not gemmwich, the screen, and the fitted operator's four products.

export module free_breakfast.cis.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.davidson;     // DavidsonOptions
export import free_breakfast.cis.ris_basis; // RisOptions and RisShells, in Options
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

/// @brief Which operator a solve diagonalizes: CIS's own A, over the
/// EriOutput's integrals, or TDA-ris's approximation of it (Zhou & Parker
/// 2024; gpu4pyscf @c tdscf/ris.py), A's two-electron part over two minimal
/// RIS fits of its own, whatever the EriOutput.
enum class Method : int32_t {
  Full, ///< A over the EriOutput's integrals, direct or fitted.
  Ris,  ///< TDA-ris: (ia|jb) over the J fit, (ij|ab) over the K fit.
};

/// @brief The knobs of one CIS solve.
struct Options {
  /// @brief The roots wanted, the lowest of A, at least 1.
  int32_t n_states = 5;
  /// @brief Singlets or triplets.
  Spin spin = Spin::Singlet;
  /// @brief The operator: CIS's A, the default, or TDA-ris's.
  Method method = Method::Full;
  /// @brief With Method::Ris, the J fit's θ: ris_basis's exponent scale for
  /// the RisShells::Sp basis the Coulomb term is fitted over (gpu4pyscf's
  /// @c theta and @c J_fit defaults). Finite and positive.
  double theta_j = 0.2;
  /// @brief With Method::Ris, the K fit's θ, for the RisShells::S basis the
  /// exchange term is fitted over (gpu4pyscf's @c K_fit default). Finite and
  /// positive.
  double theta_k = 0.2;
  /// @brief Passed to davidson_solve. Its residual_tolerance is relative,
  /// ||A x − θ x|| <= tol · max(|θ|, ||H||), and an excitation energy and the
  /// subspace's largest Ritz value are both of order one hartree, so 1e-6 is a
  /// residual of ~1e-6 hartree: an energy good to its square over the gap,
  /// ~1e-11, and a vector (so an oscillator strength) to ~1e-6 over the gap.
  DavidsonOptions davidson{.residual_tolerance = 1e-6};
  /// @brief The clustered preconditioner's gap (hartree): among the guess's
  /// lowest Δε, neighbours at most this far apart are corrected in their
  /// cluster's exact block of A (cis::Preconditioner::cluster). Zero, the
  /// default, clusters nothing: the diagonal correction alone.
  double cluster_gap = 0.0;
  /// @brief The RIS preconditioner's options (cis::RisPreconditioner): set,
  /// the correction solves (θ − Δε − 2 L Lᵀ) z = r exactly over the RIS
  /// basis it names, in place of the diagonal one, and with
  /// RisOptions::exchange refines z by GMRES towards
  /// (θ − Δε − 2 L Lᵀ + K_RI) z = r; unset, the default, it is not built.
  /// Exclusive with a nonzero @ref cluster_gap.
  std::optional<RisOptions> ris;
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

/// @brief σ = A b on the device, density-fitted in the MO basis: a
/// `LinearOperator<DfSigma, double>` over the same n_ov excitation vectors as
/// @ref Sigma, element ia at i · n_virt + a, with (pq|rs) ≈ Σ_Q B_Q,pq B_Q,rs.
///
/// It holds three MO blocks of fitted factors: B_vo over naux_j functions for
/// the Coulomb term, and B_oo and B_vv over naux_k for the exchange term. The
/// first create makes all three from one B, pipeline.eri's DensityFitted one
/// (b_mo over C's occupied and virtual columns), so naux_j = naux_k; the
/// second takes them as built, from two fits that may differ, as TDA-ris's
/// do. Then, on a whole n_ov × k block X, with w = 2 for a singlet and the
/// Coulomb term dropped for a triplet:
///
///   σ = Δε ∘ X + w · B_voᵀ (B_vo X) − Σ_Q B_oo^Q X B_vv^Q,
///
/// a dgmm, two gemms for the Coulomb term, and two strided-batched gemms for
/// the exchange term: four BLAS calls past the dgmm, however many columns.
///
/// Every buffer is allocated, and the device workspace grown to the exchange
/// term's scratch for @p max_block columns, by create; apply allocates nothing,
/// and runs a block wider than that in slices of @p max_block.
///
/// Borrows only the Δε vector, which the caller keeps alive and unchanged
/// while the sigma is in use; C and B are read only by the transforms create
/// queues on res.stream(), and not held. Move-only;
/// the buffers are on the device of the DeviceResources create was given,
/// which every apply must pass again.
class DfSigma {
public:
  /// @brief A fitted sigma over the RHF orbitals @p c (square, AO by MO,
  /// spherical, the first @p n_occ columns occupied) and the fitted factor
  /// @p b over @p naux functions (DensityFitted::b, naux × nao², [Q][n][m]),
  /// with @p delta the device Δε of cis::diagonal over the same split, for
  /// the excitations of @p spin, applying up to @p max_block columns at once.
  ///
  /// @return InvalidConfig when @p c is on another device or is not square,
  ///         when @p b does not hold naux × nao² elements, when @p delta
  ///         is null, when @p naux or @p max_block is below 1, unless
  ///         1 <= n_occ < nao, or when n_occ · n_virt, naux · n_virt or
  ///         naux · n_occ · max_block does not fit an int32_t; Unsupported,
  ///         naming the bytes, before anything is allocated, when the MO
  ///         blocks and the scratch exceed the device's free memory (an
  ///         allocation past it would abort under CreatePolicy); else
  ///         b_mo's errors.
  [[nodiscard]] static Result<DfSigma> create(DeviceResources &res,
                                              const Buffers::device<double> &b, int32_t naux,
                                              const SphericalMatrixAO &c, int32_t n_occ,
                                              const double *delta, Spin spin, int32_t max_block);

  /// @brief A fitted sigma over the MO blocks as built, taken over: @p b_vo,
  /// naux_j × n_ov with element (Q, a, i) at Q + naux_j (a + n_virt i), for
  /// the Coulomb term, and @p b_oo, naux_k × n_occ², and @p b_vv,
  /// naux_k × n_virt², with element (Q, p, q) at Q + naux_k (p + n q), for
  /// the exchange term: b_mo's layouts over (virtual, occupied),
  /// (occupied, occupied) and (virtual, virtual), on res's device. @p delta,
  /// @p spin and @p max_block are as above.
  ///
  /// @return InvalidConfig when a block does not hold its count, when
  ///         @p delta is null, when @p naux_j, @p naux_k, @p n_occ, @p n_virt
  ///         or @p max_block is below 1, or when n_occ · n_virt,
  ///         naux_k · n_virt or naux_k · n_occ · max_block does not fit an
  ///         int32_t; Unsupported, naming the bytes, when the workspace's
  ///         growth for a block exceeds the device's free memory.
  [[nodiscard]] static Result<DfSigma> create(DeviceResources &res, Buffers::device<double> b_vo,
                                              int32_t naux_j, Buffers::device<double> b_oo,
                                              Buffers::device<double> b_vv, int32_t naux_k,
                                              int32_t n_occ, int32_t n_virt, const double *delta,
                                              Spin spin, int32_t max_block);

  DfSigma(DfSigma &&) noexcept = default;
  DfSigma &operator=(DfSigma &&) noexcept = default;
  DfSigma(const DfSigma &) = delete;
  DfSigma &operator=(const DfSigma &) = delete;
  ~DfSigma() = default;

  /// @brief Y = A X for the n_ov × @p k column-major device blocks @p x and
  /// @p y, leading dimension n_ov, at most create's max_block columns per
  /// pass. Queued on res.stream() with res.blas(), whose pointer mode must be
  /// host (its default, and what davidson_solve holds); no host
  /// synchronization.
  ///
  /// @return InvalidConfig for another device or a negative @p k; else the
  ///         GpuRuntime error of the first BLAS call that failed, which leaves
  ///         @p y unspecified.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief n_ov = n_occ · n_virt, the operator's order.
  [[nodiscard]] int32_t size() const noexcept { return n_occ_ * n_virt_; }

private:
  DfSigma(DeviceResources &res, Buffers::device<double> b_vo, int32_t naux_j,
          Buffers::device<double> b_oo, Buffers::device<double> b_vv, int32_t naux_k, int32_t n_occ,
          int32_t n_virt, const double *delta, Spin spin, int32_t max_block);

  int32_t dev_idx_;
  int32_t naux_j_;
  int32_t naux_k_;
  int32_t n_occ_;
  int32_t n_virt_;
  int32_t max_block_;
  Spin spin_;
  const double *delta_;
  // State across applies, so the sigma's own: B_vo, the naux_j × n_ov matrix
  // whose column is the excitation index a + n_virt i; B_oo, [Q][i][j]; and
  // B_vv, [Q][a][b], both over naux_k.
  Buffers::device<double> b_vo_;
  Buffers::device<double> b_oo_;
  Buffers::device<double> b_vv_;
};

} // namespace free_breakfast::cis
