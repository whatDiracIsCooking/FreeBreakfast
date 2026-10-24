/// @file
/// @brief The exact orbital-orbital Hessian of a (state-averaged) CASSCF
/// wavefunction at fixed CI, over the packed non-redundant pairs: its product
/// with a block of vectors, as a LinearOperator, and its diagonal.
///
/// PySCF's mc1step.gen_g_hop h_op and h_diag, doubled as the gradient is. The
/// product is gemms over the MO matrices and (pu|vw)-type intermediates built
/// once, plus two J and two K of transition densities when there is a core;
/// see README.md for why those cannot be avoided.

export module free_breakfast.mcscf.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.mcscf.fock;

export namespace free_breakfast {

/// @brief The orbital Hessian H of E(C exp(kappa(x))) at x = 0, fixed RDMs,
/// over the packed pairs of @ref pack_orbital_pairs: E = E_0 + g . x +
/// 1/2 x^T H x + ..., with g @ref CasscfFock::gradient.
///
/// A `LinearOperator<OrbitalHessian, double>` of order size(), so
/// davidson_solve (or an augmented Hessian over it) takes it as it is, and
/// diagonal() is the exact diag(H) for the preconditioner. H is symmetric.
///
/// Holds the MO intermediates it was built from, so the orbitals, ppaa, papa
/// and the CasscfFock may go once create returns; it keeps @p j_classes and
/// @p k_classes (by span), which must outlive it, and the three Cartesian
/// matrices J and K read and write. Move-only.
class OrbitalHessian {
public:
  /// @brief Build H at the orbitals @p c and the weighted RDMs in @p fock.
  ///
  /// @p basis, @p h, @p u and the classes are @ref casscf_fock's; @p fock is
  /// casscf_fock's result over the same @p c, @p ncore and @p ncas, and
  /// @p ppaa and @p papa are @c casscf_integrals' over the same @p c (both
  /// empty for ncas = 0). Each is read here and not kept, the classes
  /// excepted.
  ///
  /// It forms C^T h C, the (pu|rv)-type matrix H_act = sum_wx (pr|wx)
  /// Gamma_wxuv + (pw|rx) (Gamma_uwxv + Gamma_wuxv) over the active columns
  /// (nmo ncas)^2 doubles, by strided-batched gemms over @p ppaa and @p papa,
  /// and the diagonal, which takes (pp|ii) and (pi|pi) over every p and core
  /// i: one @c coulomb and one @c exchange of C_i C_i^T per core orbital, each
  /// under the |D| <= 2 prune cap as @c casscf_integrals holds it.
  ///
  /// Queued on res.stream() with res.blas() (host pointer mode); it
  /// synchronizes res.stream() to read back the RDMs and the diagonals the
  /// diagonal is assembled from, and once at the end. Its scratch is
  /// res.device_workspace()'s. The device res.dev_idx() names is made current
  /// for the call and the previous one restored.
  ///
  /// @return The operator; InvalidConfig if ncore or ncas is negative,
  ///         ncore + ncas is 0 or exceeds nmo, a tau is negative or NaN, @p h
  ///         or @p c is not of @p basis's spherical order or not on res's
  ///         device, @p ppaa or @p papa is not nmo^2 ncas^2 long, or a member
  ///         of @p fock has another size than these ncore and ncas give; or
  ///         the error of the first J, K, BLAS call or copy that failed.
  [[nodiscard]] static Result<OrbitalHessian>
  create(DeviceResources &res, const BasisSet &basis, const SphericalMatrixAO &h,
         const std::shared_ptr<const Buffers::device<double>> &u,
         std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
         const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
         const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
         const CasscfFock &fock, double tau_j = 1e-14, double tau_k = 1e-14);

  OrbitalHessian(OrbitalHessian &&) noexcept = default;
  OrbitalHessian &operator=(OrbitalHessian &&) noexcept = default;
  OrbitalHessian(const OrbitalHessian &) = delete;
  OrbitalHessian &operator=(const OrbitalHessian &) = delete;
  ~OrbitalHessian() = default;

  /// @brief Y = H X for the size() x @p k column-major device blocks @p x and
  /// @p y, leading dimension size(); @p x and @p y must not overlap.
  ///
  /// Per column: kappa(x) unpacked, the MO gemms and one gemv over H_act, and
  /// for ncore > 0 one @c coulomb and one @c exchange of each of two
  /// transition densities, each under the prune cap (its max |D| read back
  /// through iamax, which synchronizes); then 2 (X2 - X2^T) packed. Queued on
  /// res.stream() with res.blas(), whose pointer mode must be host (its
  /// default, and what davidson_solve holds). Its scratch is
  /// res.device_workspace()'s. The device res.dev_idx() names is made current
  /// for the call and the previous one restored.
  ///
  /// @return InvalidConfig for another device than create's or a negative
  ///         @p k; or the error of the first J, K, BLAS call or copy that
  ///         failed, which leaves @p y unspecified.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief The order of H: ncore (nmo - ncore) + ncas (nmo - ncore - ncas).
  [[nodiscard]] int32_t size() const noexcept { return n_pairs_; }

  /// @brief diag(H) over the packed pairs, on the device: size() doubles.
  [[nodiscard]] const Buffers::device<double> &diagonal() const noexcept { return diagonal_; }

private:
  OrbitalHessian(DeviceResources &res, const BasisSet &basis, std::span<CoulombPairClass> j_classes,
                 std::span<OrderedPairClass> k_classes, int32_t nmo, int32_t ncore, int32_t ncas,
                 double tau_j, double tau_k);

  // max |D| of the n_cart x n_cart @p from read back, and @p to = s @p from
  // with s = min(1, 2 / max |D|): the |D| <= 2 the J classes prune for.
  [[nodiscard]] Result<double> capped_density(DeviceResources &res, const double *from);

  std::span<CoulombPairClass> j_classes_;
  std::span<OrderedPairClass> k_classes_;
  int32_t dev_idx_;
  int32_t nmo_;
  int32_t ncore_;
  int32_t ncas_;
  int32_t n_cart_;
  int32_t n_pairs_;
  double tau_j_;
  double tau_k_;
  Buffers::device<double> c_cart_;   // U C, n_cart x nmo
  Buffers::device<double> h_mo_;     // C^T h C, nmo x nmo
  Buffers::device<double> fock_sym_; // F + F^T, nmo x nmo
  Buffers::device<double> vhf_c_;    // F^I - C^T h C
  Buffers::device<double> vhf_ca_;   // F^I + F^A - C^T h C
  Buffers::device<double> dm1_occ_;  // the 1-RDM's occupied block, nocc x nocc
  Buffers::device<double> gamma_;    // gamma, ncas x ncas
  Buffers::device<double> h_act_;    // (nmo ncas)^2, (p, u) at p + nmo u
  Buffers::device<double> diagonal_; // diag(H), n_pairs
  CartesianMatrixAO density_;
  CartesianMatrixAO j_;
  CartesianMatrixAO k_;
};

} // namespace free_breakfast
