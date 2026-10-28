/// @file
/// @brief The exact orbital-orbital Hessian of a (state-averaged) CASSCF
/// wavefunction at fixed CI, over the packed non-redundant pairs: its product
/// with a block of vectors, as a LinearOperator, and its diagonal.
///
/// PySCF's mc1step.gen_g_hop h_op and h_diag, doubled as the gradient is. The
/// product is gemms over the MO matrices and (pu|vw)-type intermediates built
/// once, plus two J and two K of transition densities when there is a core,
/// integral-direct or from the fitted factor B (PySCF's approx_hessian); see
/// README.md for why those cannot be avoided.

export module free_breakfast.response.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.response.fock;

export namespace free_breakfast::response {

/// @brief The orbital Hessian H of E(C exp(kappa(x))) at x = 0, fixed RDMs,
/// over the packed pairs of @ref pack_orbital_pairs: E = E_0 + g . x +
/// 1/2 x^T H x + ..., with g @ref Fock::gradient.
///
/// A `LinearOperator<Hessian, double>` of order size(), so
/// davidson_solve (or an augmented Hessian over it) takes it as it is, and
/// diagonal() is the exact diag(H) for the preconditioner. H is symmetric to
/// 100 tau_k max |H| |x| |y| (README.md).
///
/// Holds the MO intermediates it was built from, so the orbitals, ppaa, papa
/// and the Fock may go once create returns. Its J and K are of one of two
/// kinds, chosen by the create called: integral-direct, keeping @p j_classes
/// and @p k_classes (by span) and the three Cartesian matrices J and K read
/// and write; or density-fitted, keeping @p b's storage and two nmo x nmo
/// matrices. What it keeps must outlive it. Move-only.
class Hessian {
public:
  /// @brief Build H at the orbitals @p c and the weighted RDMs in @p fock.
  ///
  /// @p basis, @p h, @p u and the classes are @ref fock's; @p fock is
  /// fock's result over the same @p c, @p ncore and @p ncas, and
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
  [[nodiscard]] static Result<Hessian>
  create(DeviceResources &res, const BasisSet &basis, const SphericalMatrixAO &h,
         const std::shared_ptr<const Buffers::device<double>> &u,
         std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
         const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
         const Buffers::device<double> &ppaa, const Buffers::device<double> &papa, const Fock &fock,
         double tau_j = 1e-14, double tau_k = 1e-14);

  /// @brief The integral-direct @ref create's H, density-fitted: J and K
  /// from the fitted factor @p b in place of the J and K classes, the rest
  /// as there.
  ///
  /// @p b is B over @p naux fitting functions and C's nmo = h.n() spherical
  /// AOs, as @c df_coulomb reads it (@c DensityFittingOutput::b): naux nmo^2
  /// doubles, whose storage is kept (by pointer, so the buffer may be moved
  /// but not freed) for every @ref apply. @p h, @p c, @p ppaa,
  /// @p papa and @p fock are as there; @p ppaa, @p papa and @p fock are
  /// whatever the caller fitted them by, so H is wholly density-fitted only
  /// when they are.
  ///
  /// The diagonal's (pp|ii) and (pi|pi) come from L_Q,ip = (Q|ip), B over
  /// C_core and C: one batched gemm and one gemm, then (Q|pp) by batched
  /// gemvs over B half-transformed ncore orbitals at a time, a gemm for
  /// (pp|ii) and batched dots for (pi|pi). No J or K per core orbital, no
  /// U C, no screen and no prune cap. Its scratch is res.device_workspace()'s:
  /// 2 naux nmo ncore + naux nmo + 2 nmo ncore doubles past the
  /// integral-direct create's.
  ///
  /// Queued and synchronized as the integral-direct @ref create.
  ///
  /// @return The operator; InvalidConfig as the integral-direct @ref create's
  ///         (but the taus' and the basis'), and if naux is negative, @p c is
  ///         not h.n() x h.n(), or @p b is not naux nmo^2 long; or the error
  ///         of the first BLAS call or copy that failed.
  [[nodiscard]] static Result<Hessian> create(DeviceResources &res, const SphericalMatrixAO &h,
                                              const Buffers::device<double> &b, int32_t naux,
                                              const SphericalMatrixAO &c, int32_t ncore,
                                              int32_t ncas, const Buffers::device<double> &ppaa,
                                              const Buffers::device<double> &papa,
                                              const Fock &fock);

  Hessian(Hessian &&) noexcept = default;
  Hessian &operator=(Hessian &&) noexcept = default;
  Hessian(const Hessian &) = delete;
  Hessian &operator=(const Hessian &) = delete;
  ~Hessian() = default;

  /// @brief Y = H X for the size() x @p k column-major device blocks @p x and
  /// @p y, leading dimension size(); @p x and @p y must not overlap.
  ///
  /// Per column: kappa(x) unpacked, the MO gemms and one gemv over H_act, and
  /// for ncore > 0 a J and a K of each of two transition densities; then
  /// 2 (X2 - X2^T) packed. Integral-direct, each is one @c coulomb and one
  /// @c exchange under the prune cap (its max |D| read back through
  /// norm<Norm::inf>, which synchronizes), and the scratch is
  /// res.device_workspace()'s. Density-fitted, each is one @c df_coulomb of
  /// the density and one two-factor @c df_exchange of its factors, with no
  /// prune cap and no synchronize; those take the workspace, so the scratch
  /// is allocated per call. Queued on res.stream() with res.blas(), whose
  /// pointer mode must be host (its default, and what davidson_solve holds).
  /// The device res.dev_idx() names is made current for the call and the
  /// previous one restored.
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
  // Integral-direct J and K: the classes and their screens, and the three
  // Cartesian matrices coulomb and exchange read and write.
  struct Direct {
    std::span<CoulombPairClass> j_classes;
    std::span<OrderedPairClass> k_classes;
    double tau_j;
    double tau_k;
    CartesianMatrixAO density;
    CartesianMatrixAO j;
    CartesianMatrixAO k;
  };
  // Density-fitted J and K: B's storage, the caller's, and the nmo x nmo
  // matrices df_coulomb and df_exchange write.
  struct Fitted {
    const double *b;
    int32_t naux;
    Buffers::device<double> j;
    Buffers::device<double> k;
  };

  Hessian(DeviceResources &res, int32_t nmo, int32_t ncore, int32_t ncas, int32_t n_cart,
          std::variant<Direct, Fitted> jk);

  // Both creates past their own checks: U C (U null for none) has n_cart_rows
  // rows.
  [[nodiscard]] static Result<Hessian>
  build(DeviceResources &res, const SphericalMatrixAO &h,
        const std::shared_ptr<const Buffers::device<double>> &u, const SphericalMatrixAO &c,
        int32_t ncore, int32_t ncas, const Buffers::device<double> &ppaa,
        const Buffers::device<double> &papa, const Fock &fock, int32_t n_cart_rows,
        std::variant<Direct, Fitted> jk);

  // G = 2 J[D] - K[D] of the n_cart x n_cart D = @p density, D = L R^T + R L^T
  // with L the first @p rank columns of U C and R = @p right (n_cart x rank):
  // returns s G and s. Integral-direct, s = min(1, 2 / max |D|), the |D| <= 2
  // the J classes prune for, and @p right is not read; density-fitted, s = 1
  // and @p density is J's, the factors K's.
  [[nodiscard]] Result<std::pair<const double *, double>>
  two_j_minus_k(DeviceResources &res, const double *density, const double *right, int32_t rank);

  std::variant<Direct, Fitted> jk_;
  int32_t dev_idx_;
  int32_t nmo_;
  int32_t ncore_;
  int32_t ncas_;
  int32_t n_cart_;
  int32_t n_pairs_;
  Buffers::device<double> c_cart_;   // U C, n_cart x nmo (C, density-fitted)
  Buffers::device<double> h_mo_;     // C^T h C, nmo x nmo
  Buffers::device<double> fock_sym_; // F + F^T, nmo x nmo
  Buffers::device<double> vhf_c_;    // F^I - C^T h C
  Buffers::device<double> vhf_ca_;   // F^I + F^A - C^T h C
  Buffers::device<double> dm1_occ_;  // the 1-RDM's occupied block, nocc x nocc
  Buffers::device<double> gamma_;    // gamma, ncas x ncas
  Buffers::device<double> h_act_;    // (nmo ncas)^2, (p, u) at p + nmo u
  Buffers::device<double> diagonal_; // diag(H), n_pairs
};

} // namespace free_breakfast::response
