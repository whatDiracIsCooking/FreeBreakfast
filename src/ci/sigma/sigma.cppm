/// @file
/// @brief σ = Hc and diag(H) for a determinant CI vector over two StringSets,
/// from the active-space integrals as plain arrays: CiSigma on the device, as
/// a LinearOperator, with the 1-RDM over the same tables, and host_sigma /
/// host_diagonal on the host.
///
/// The host pair is written for clarity rather than speed: it is the reference
/// the GPU sigma is tested against.

export module free_breakfast.ci.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;

export namespace free_breakfast {

/// @brief How a device CI vector is laid out.
///
/// full is host_sigma's: every determinant, I_α · beta.size() + I_β. symmetric
/// holds a vector over one StringSet for both spins (N_α = N_β) that is
/// symmetric under the swap of the two strings, c(I_α, I_β) = c(I_β, I_α), as
/// every state of even S is: only the triangle I_α ≥ I_β, element
/// I_α (I_α + 1) / 2 + I_β, each off-diagonal one (I_α > I_β) scaled by √2.
/// The scaling makes the packing an isometry, so the plain dot product, and
/// with it a symmetric operator's symmetry, are kept, and davidson_solve runs
/// on packed vectors as it is. A set of m strings packs to m (m + 1) / 2.
enum class CiLayout : int32_t { full, symmetric };

/// @brief Unpack the @p k symmetric-layout device vectors @p x, each
/// m (m + 1) / 2 long for @p n_strings = m, into the @p k full-layout ones
/// @p y, each m², both blocks column-major. Element (I_α, I_β) of y is x's
/// triangle element (max, min), divided by √2 off the diagonal. Queued on
/// res.stream().
///
/// @return InvalidConfig for a negative @p n_strings or @p k; or the launch's
///         GpuRuntime error.
[[nodiscard]] Status ci_unpack(DeviceResources &res, int32_t n_strings, int32_t k, const double *x,
                               double *y);

/// @brief Pack the @p k full-layout device vectors @p x, each m² long for
/// @p n_strings = m, into the @p k symmetric-layout ones @p y, each
/// m (m + 1) / 2, both blocks column-major: ci_unpack's inverse on vectors
/// symmetric under the swap of the two strings. Triangle element (I_α, I_β)
/// of y is (x(I_α, I_β) + x(I_β, I_α)) / √2 off the diagonal and x(I_α, I_α)
/// on it, which is ci_unpack's adjoint: any x goes to the packing of its
/// swap-symmetric part. Queued on res.stream().
///
/// @return InvalidConfig for a negative @p n_strings or @p k; or the launch's
///         GpuRuntime error.
[[nodiscard]] Status ci_pack(DeviceResources &res, int32_t n_strings, int32_t k, const double *x,
                             double *y);

/// @brief σ = H c on the host, H = E_core + Σ_tu h_tu E_tu
/// + ½ Σ_tuvw (tu|vw) (E_tu E_vw − δ_uv E_tw), E_tu the spin-summed
/// excitation operator.
///
/// The determinants are the pairs (I_α, I_β) of @p alpha's and @p beta's
/// strings, laid out alpha-major: element I_α · beta.size() + I_β, which is
/// `fci.direct_spin1`'s layout. The α–α, β–β and α–β terms all go through the
/// two sets' replacement lists. Cost is O(dim · n⁴), memory
/// O(n²) besides the vectors.
///
/// @param alpha, beta The strings of each spin, over the same n orbitals.
/// @param e_core The constant: nuclear repulsion and the frozen core's energy.
/// @param h_eff n × n, row-major, symmetric: the one-electron operator over
///        the active orbitals, the core's mean field included.
/// @param eri n⁴, row-major (t, u, v, w), chemists' order, real with the
///        8-fold symmetry: (tu|vw).
/// @param c The CI vector, alpha.size() · beta.size().
/// @param sigma Written with H c, the same size; must not overlap @p c.
/// @return InvalidConfig when the two sets span different orbital counts or
///         any span has the wrong size; nothing is written then.
[[nodiscard]] Status host_sigma(const StringSet &alpha, const StringSet &beta, double e_core,
                                std::span<const double> h_eff, std::span<const double> eri,
                                std::span<const double> c, std::span<double> sigma);

/// @brief diag(H) on the host, for the same H and layout as host_sigma.
///
/// Slater–Condon's rule for a determinant's diagonal over its occupied spin
/// orbitals, α (ascending) then β (ascending), i < j:
/// E_core + Σ_i h_ii + Σ_{i<j} [(ii|jj) − δ_{σ_i σ_j} (ij|ji)], summed in that
/// order.
///
/// @param diagonal Written with H_II, alpha.size() · beta.size().
/// @return InvalidConfig as host_sigma does.
[[nodiscard]] Status host_diagonal(const StringSet &alpha, const StringSet &beta, double e_core,
                                   std::span<const double> h_eff, std::span<const double> eri,
                                   std::span<double> diagonal);

/// @brief σ = Hc and diag(H) on the device, for host_sigma's H and layout: a
/// `LinearOperator<CiSigma, double>` over blocks of CI vectors.
///
/// create uploads the two sets' strings and replacement lists, once per
/// (n, N_α, N_β); set_integrals uploads an H over them, and may be called again
/// for another (a CASSCF macro-iteration's) without touching the tables.
///
/// apply gathers D_K[vw] = (E_vw c)_K over each determinant's replacement
/// lists, contracts it with the integrals in one gemm per chunk of rows,
/// G_K = D_K [½ (tu|vw) | k_vw], and gathers σ_I from G over I's own lists:
/// both gathers are one thread per determinant, so nothing is atomic and the
/// result does not depend on the run. D and G are packed over the orbital
/// pairs v ≥ w, since (tu|vw) is symmetric in each pair. The block goes one
/// vector at a time, so its scratch, G for one vector (dim · (n(n+1)/2 + 1)
/// doubles) and D for one chunk, is res.device_workspace() and does not grow
/// with the block.
///
/// Over one StringSet for both spins, apply and apply_fp32 also take blocks in
/// CiLayout::symmetric, where H's commuting with the swap of the two strings
/// lets only the rows I_α ≥ I_β be computed: G_J for J below the triangle is
/// G of its mirror. That halves the gathers, the gemm and G's scratch.
///
/// G for one vector can still be past the device (5 GB at CAS(14,14) in the
/// symmetric layout), so set_workspace_limit bounds the scratch: past the
/// limit, G is computed a block of strings at a time, at up to twice the
/// gathers and gemm.
///
/// Move-only; the buffers are on the device of the DeviceResources create was
/// given, which every call must pass again.
class CiSigma {
public:
  /// @brief Upload @p alpha's and @p beta's strings and replacement lists to
  /// @p res's device. A failed allocation aborts (CreatePolicy).
  ///
  /// @return InvalidConfig when the two sets span different orbital counts or
  ///         none, or alpha.size() · beta.size() does not fit an int32_t; or the
  ///         GpuRuntime error of an upload.
  [[nodiscard]] static Result<CiSigma> create(DeviceResources &res, const StringSet &alpha,
                                              const StringSet &beta);

  CiSigma(CiSigma &&) noexcept = default;
  CiSigma &operator=(CiSigma &&) noexcept = default;
  CiSigma(const CiSigma &) = delete;
  CiSigma &operator=(const CiSigma &) = delete;
  ~CiSigma() = default;

  /// @brief Upload the H of host_sigma over these strings: @p e_core, @p h_eff
  /// (n × n) and @p eri (n⁴, (tu|vw)), host arrays with host_sigma's layout and
  /// symmetry. Synchronizes res.stream() before returning, so the arrays may
  /// go at once.
  ///
  /// @return InvalidConfig for another device or a span of the wrong size,
  ///         leaving the previous integrals in place; or the GpuRuntime error
  ///         of an upload.
  [[nodiscard]] Status set_integrals(DeviceResources &res, double e_core,
                                     std::span<const double> h_eff, std::span<const double> eri);

  /// @brief Y = H X for the size(@p layout) × @p k column-major device blocks
  /// @p x and @p y, leading dimension size(@p layout). Queued on res.stream()
  /// with res.blas(), whose pointer mode must be host (its default, and what
  /// davidson_solve holds).
  ///
  /// CiLayout::symmetric takes and gives vectors symmetric under the swap of
  /// the two strings, packed (see CiLayout); its scratch is G for the
  /// triangle's rows, half the full layout's.
  ///
  /// @return InvalidConfig before set_integrals, for another device, for a
  ///         negative @p k, or for CiLayout::symmetric over two different
  ///         sets; or the GpuRuntime error of a launch or the gemm.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y,
                             CiLayout layout = CiLayout::full);

  /// @brief apply in single precision: the same Y = H X over the same double
  /// blocks, but D, G and the contraction matrix are float and the gemm is
  /// gemm<float>, which halves the gathers' traffic and G's scratch and runs
  /// at the FP32 rate. Each D entry is the float of at most two FP64 terms,
  /// and the second gather sums G in double, so the error is float rounding's,
  /// of D, the contraction and G: a column of Y is good to ~1e-7 relative of
  /// ‖Y − E_core X‖, not to roundoff. ci_solve's first stage, never its
  /// converged roots.
  ///
  /// @return As apply.
  [[nodiscard]] Status apply_fp32(DeviceResources &res, int32_t k, const double *x, double *y,
                                  CiLayout layout = CiLayout::full);

  /// @brief Bound the device workspace apply and apply_fp32 take to @p bytes;
  /// unbounded until called.
  ///
  /// Where G for a whole vector and a chunk of D fit, a call runs as it would
  /// unbounded, and gives the same result to the bit. Past it, G is held for a
  /// block of strings at a time, as many as fit: in the full layout α blocks,
  /// whose G serves the β links of their own rows, then β blocks, serving the
  /// α links; in the symmetric layout one pass of blocks over the one set
  /// serves both (see the README). That runs the first gather and the gemm
  /// over every determinant twice in the full layout, and over all m² rather
  /// than the triangle's in the symmetric one, and agrees with the unblocked
  /// result to roundoff: σ_I sums the same terms in another order. A block is
  /// never less than one string, so a limit under one string's G and D is
  /// exceeded by that much.
  ///
  /// While bounded, a call first grows the workspace to the lesser of
  /// @p bytes and apply's unblocked scratch, so apply_fp32 followed by apply
  /// (ci_solve's two stages) shares one allocation rather than holding the old
  /// and the new across a growth: the bound covers the peak.
  void set_workspace_limit(size_t bytes) noexcept { workspace_limit_ = bytes; }

  /// @brief diag(H) into the size() device doubles @p diagonal, by
  /// host_diagonal's rule and in its order. Queued on res.stream().
  ///
  /// @return InvalidConfig before set_integrals or for another device; or the
  ///         launch's GpuRuntime error.
  [[nodiscard]] Status diagonal(DeviceResources &res, double *diagonal);

  /// @brief The spin-summed 1-RDM γ_tu = ⟨c| E_tu |c⟩ of the size(@p layout)
  /// device doubles @p c into the n × n device doubles @p gamma (symmetric, so
  /// either order): ci.rdm's rdm1, on the device. Needs no integrals.
  ///
  /// apply's gather of D_K = (E c)_K over each chunk of determinants, packed
  /// over v ≥ w, then one gemv per chunk, Σ_K c_K D_K, which is 2 γ_vw off the
  /// diagonal and γ_vv on it, unpacked into @p gamma. Its scratch, D for one
  /// chunk and the n(n+1)/2 packed sums, is res.device_workspace(). Queued on
  /// res.stream() with res.blas(), in host pointer mode.
  ///
  /// CiLayout::symmetric takes a packed swap-symmetric c, for which D_K is D
  /// of K's mirror: the gather walks the triangle's rows alone, each
  /// off-diagonal one scaled by √2, and with the packed c's own √2 the gemv's
  /// sum gives each the weight 2 of itself and its mirror. That halves the
  /// gather and the gemv.
  ///
  /// @return InvalidConfig for another device, or for CiLayout::symmetric over
  ///         two different sets; or the GpuRuntime error of a launch or a gemv.
  [[nodiscard]] Status rdm1(DeviceResources &res, const double *c, double *gamma,
                            CiLayout layout = CiLayout::full);

  /// @brief The operator's n: the number of determinants,
  /// alpha.size() · beta.size(), or for CiLayout::symmetric the triangle's,
  /// alpha.size() (alpha.size() + 1) / 2.
  [[nodiscard]] int32_t size(CiLayout layout = CiLayout::full) const noexcept {
    return layout == CiLayout::symmetric ? n_alpha_ * (n_alpha_ + 1) / 2 : n_alpha_ * n_beta_;
  }

private:
  CiSigma(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
          int32_t alpha_per, int32_t beta_per, bool one_set);

  // apply and apply_fp32: D and G of T, over the contraction of T.
  template <typename T>
  Status apply_as(DeviceResources &res, int32_t k, const double *x, double *y, const T *contraction,
                  CiLayout layout);

  // apply_as past workspace_limit_: G a block of strings at a time.
  template <typename T>
  Status apply_blocked(DeviceResources &res, int32_t k, const double *x, double *y,
                       const T *contraction, CiLayout layout);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  int32_t beta_per_;
  bool one_set_; // alpha and beta are the same (n, N): CiLayout::symmetric applies
  bool has_integrals_ = false;
  double e_core_ = 0.0;
  // set_workspace_limit's bound on apply's scratch, in bytes.
  size_t workspace_limit_ = std::numeric_limits<size_t>::max();
  // The link tables, three int32_t per replacement: target, packed pair, sign.
  Buffers::device<int32_t> alpha_links_;
  Buffers::device<int32_t> beta_links_;
  Buffers::device<uint64_t> alpha_strings_;
  Buffers::device<uint64_t> beta_strings_;
  // n_pairs × (n_pairs + 1), column-major: column pair(t, u) is ½ (tu|vw) over
  // the rows pair(v, w), and the last column k_vw.
  Buffers::device<double> contraction_;
  // contraction_ rounded to float, apply_fp32's.
  Buffers::device<float> contraction_fp32_;
  // n + 2 n²: h_ii, (ii|jj) and (ij|ji), for the diagonal.
  Buffers::device<double> diagonal_terms_;
};

} // namespace free_breakfast
