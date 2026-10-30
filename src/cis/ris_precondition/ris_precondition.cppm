/// @file
/// @brief The RIS preconditioner (Zhou & Parker 2024; gpu4pyscf
/// @c tdscf/ris.py): CIS's A approximated as Δε + 2 L Lᵀ over the RIS
/// minimal auxiliary basis (RIS-J), its shifted system solved exactly per
/// root by Sherman–Morrison–Woodbury, and optionally the RI exchange −K_RI
/// added (RIS-JK), solved by a short GMRES preconditioned by that solve; as a
/// DavidsonPreconditioner.
///
/// An element ia of an n_ov vector is i · n_virt + a: row-major (i, a). The
/// README has the derivation, the clamp, the singular fallback and the
/// GMRES's.

export module free_breakfast.cis.ris_precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.integrals.mmd; // PrimitivePairClass
import free_breakfast.recipes.matrix_ao;
import free_breakfast.cis.ris_basis; // RisOptions

export namespace free_breakfast::cis {

/// @brief The RIS-J correction as a DavidsonPreconditioner: per column c,
/// z solving (θ_c − Δε − 2 L Lᵀ) z = r exactly, with L_ia,Q = B_Q,ia the
/// fitted factor over the RIS basis in the ov block, so that
/// L Lᵀ = (ia|P) M⁻¹ (P|jb).
///
/// Per column, with Dt = θ_c − Δε under @ref Preconditioner's clamp (its
/// magnitude held at or above 1e-3, its sign kept):
///
///   z = Dt⁻¹ (r + L y),   (½ I − Lᵀ Dt⁻¹ L) y = Lᵀ Dt⁻¹ r,
///
/// the naux × naux capacitance system, symmetric and indefinite, solved by
/// @ref lu_solve. A column whose capacitance matrix is exactly singular takes
/// y = 0: the diagonal correction r / Dt.
///
/// With RisOptions::exchange (RIS-JK), A ≈ Δε + 2 L Lᵀ − K_RI, with
/// (K_RI)_ia,jb = Σ_Q T_ij^Q T_ab^Q over the same fit's oo and vv blocks.
/// Writing M = Dt − 2 L Lᵀ, each column's z then solves
/// (I + M⁻¹ K_RI) z = M⁻¹ r by unrestarted GMRES from z₀ = M⁻¹ r, the RIS-J
/// answer, for at most RisOptions::k_max_inner_iters steps or until the
/// residual is RisOptions::k_inner_tol of ‖z₀‖. A breakdown before any
/// progress, a singular triangle or a value that is not finite leaves z at
/// z₀; a cap reached first keeps GMRES's iterate, whose residual is no
/// larger than z₀'s.
///
/// Every buffer a call writes is allocated by create, the scratch it holds
/// across lu_solve (which takes the device workspace) among them, so a call
/// allocates only what lu_solve's workspace may grow by. @p delta is borrowed,
/// not copied: the caller keeps it alive and unchanged while this is in use.
/// Move-only; the buffers are on the device of the DeviceResources create was
/// given, which every call must pass again.
class RisPreconditioner {
public:
  /// @brief The RIS-J preconditioner over the RHF orbitals @p c (square, AO
  /// by MO, spherical, the first @p n_occ columns occupied) of @p basis, with
  /// @p delta the device Δε of cis::diagonal over the same split, for calls
  /// of at most @p max_columns columns.
  ///
  /// The RIS basis (@ref ris_basis over @p atoms at @p ris) is fitted as
  /// pipeline.eri fits its own, whatever that one is: its metric, the
  /// three-centre (mn|P) over @p classes (primitive_pair_classes of @p basis)
  /// and density_fit, B = L_M⁻¹ (mn|P); then L is b_mo of B over the virtual
  /// and occupied columns of @p c, and with ris.exchange T_oo and T_vv are
  /// b_mo over the occupied, and the virtual, columns on both sides. B is
  /// freed on return.
  ///
  /// @return InvalidConfig when @p c is on another device or is not
  ///         basis.nao_sphe() square, when @p delta is null, when
  ///         @p max_columns is below 1, unless 1 <= n_occ < nao with
  ///         n_occ · n_virt fitting an int32_t, or, with ris.exchange, for a
  ///         negative ris.k_max_inner_iters, a ris.k_inner_tol that is not
  ///         positive, or naux · n_virt past an int32_t; Unsupported, naming
  ///         the bytes, before anything is allocated, when B, L's two copies
  ///         and the exchange's blocks and GMRES scratch exceed the device's
  ///         free memory; else the error of ris_basis, metric, three_center,
  ///         density_fit or b_mo.
  [[nodiscard]] static Result<RisPreconditioner>
  create(DeviceResources &res, const BasisSet &basis, std::span<const PrimitivePairClass> classes,
         std::span<const Atom> atoms, const SphericalMatrixAO &c, int32_t n_occ,
         const double *delta, int32_t max_columns, RisOptions ris);

  RisPreconditioner(RisPreconditioner &&) noexcept = default;
  RisPreconditioner &operator=(RisPreconditioner &&) noexcept = default;
  RisPreconditioner(const RisPreconditioner &) = delete;
  RisPreconditioner &operator=(const RisPreconditioner &) = delete;
  ~RisPreconditioner() = default;

  /// @brief correction = (θ − Δε − 2 L Lᵀ)⁻¹ residual, column by column, for
  /// the theta.size() n_ov-long columns of the device blocks, refined
  /// towards (θ − Δε − 2 L Lᵀ + K_RI)⁻¹ residual with exchange;
  /// @p ritz_vectors is not read. @p correction may be @p residual. Queued on
  /// res.stream() with res.blas() in host pointer mode (its default);
  /// synchronizes twice per column in lu_solve, and with exchange five times
  /// more per column and twice per GMRES step.
  ///
  /// @return InvalidConfig for more than max_columns columns or another
  ///         device; else the GpuRuntime error of the first call that failed.
  [[nodiscard]] Status operator()(DeviceResources &res, std::span<const double> theta,
                                  const double *ritz_vectors, const double *residual,
                                  double *correction) const;

  /// @brief The RIS basis's function count, L's columns.
  [[nodiscard]] int32_t naux() const noexcept { return naux_; }

  /// @brief Lᵀ, the device naux × n_ov matrix whose column i · n_virt + a is
  /// excitation ia's row of L, B_Q,ia.
  [[nodiscard]] const Buffers::device<double> &factor() const noexcept { return factor_; }

  /// @brief T_oo, the device naux × n_occ² block whose element
  /// Q + naux (i + n_occ j) is T_ij^Q; empty without exchange.
  [[nodiscard]] const Buffers::device<double> &factor_oo() const noexcept { return factor_oo_; }

  /// @brief T_vv, the device naux × n_virt² block whose element
  /// Q + naux (a + n_virt b) is T_ab^Q; empty without exchange.
  [[nodiscard]] const Buffers::device<double> &factor_vv() const noexcept { return factor_vv_; }

private:
  RisPreconditioner(DeviceResources &res, int32_t n_occ, int32_t n_virt, int32_t naux,
                    const double *delta, int32_t max_columns, RisOptions ris);

  // Column col's Dt⁻¹ into inverse_, Lᵀ Dt⁻¹ into scaled_ and the
  // capacitance matrix ½ I − Lᵀ Dt⁻¹ L into capacitance_, theta_ uploaded.
  [[nodiscard]] Status build_capacitance(DeviceResources &res, int32_t col) const;

  // Column col of the correction, z, refined by the exchange's GMRES from the
  // RIS-J answer it holds.
  [[nodiscard]] Status refine(DeviceResources &res, int32_t col, double *z) const;

  int32_t dev_idx_;
  int32_t n_occ_;
  int32_t n_virt_;
  int32_t n_ov_;
  int32_t naux_;
  bool exchange_;
  int32_t k_max_inner_iters_;
  double k_inner_tol_;
  const double *delta_;
  Buffers::device<double> factor_;
  Buffers::device<double> factor_oo_;
  Buffers::device<double> factor_vv_;
  // A vector of ones, from which the precondition kernel makes Dt⁻¹, and the
  // ½ I the capacitance matrix starts from.
  Buffers::device<double> ones_;
  Buffers::device<double> half_identity_;
  // The concept calls the preconditioner const; these are per-call scratch,
  // held across lu_solve, which takes the device workspace, so the
  // preconditioner's own: θ, Dt⁻¹ for one column, Lᵀ Dt⁻¹ (naux × n_ov), the
  // capacitance matrix, and the right-hand sides y, one column each.
  mutable Buffers::device<double> theta_;
  mutable Buffers::device<double> inverse_;
  mutable Buffers::device<double> scaled_;
  mutable Buffers::device<double> capacitance_;
  mutable Buffers::device<double> y_;
  // With exchange, ½ C⁻¹ of one column's capacitance matrix C: lu_solve's
  // right-hand side, so not the workspace's. Empty without.
  mutable Buffers::device<double> half_inverse_;
};

} // namespace free_breakfast::cis
