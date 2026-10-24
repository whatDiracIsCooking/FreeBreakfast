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

  /// @brief Y = H X for the size() × @p k column-major device blocks @p x and
  /// @p y, leading dimension size(). Queued on res.stream() with res.blas(),
  /// whose pointer mode must be host (its default, and what davidson_solve
  /// holds).
  ///
  /// @return InvalidConfig before set_integrals, for another device or for a
  ///         negative @p k; or the GpuRuntime error of a launch or the gemm.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief diag(H) into the size() device doubles @p diagonal, by
  /// host_diagonal's rule and in its order. Queued on res.stream().
  ///
  /// @return InvalidConfig before set_integrals or for another device; or the
  ///         launch's GpuRuntime error.
  [[nodiscard]] Status diagonal(DeviceResources &res, double *diagonal);

  /// @brief The spin-summed 1-RDM γ_tu = ⟨c| E_tu |c⟩ of the size() device
  /// doubles @p c into the n × n device doubles @p gamma (symmetric, so either
  /// order): ci.rdm's rdm1, on the device. Needs no integrals.
  ///
  /// apply's gather of D_K = (E c)_K over each chunk of determinants, packed
  /// over v ≥ w, then one gemv per chunk, Σ_K c_K D_K, which is 2 γ_vw off the
  /// diagonal and γ_vv on it, unpacked into @p gamma. Its scratch, D for one
  /// chunk and the n(n+1)/2 packed sums, is res.device_workspace(). Queued on
  /// res.stream() with res.blas(), in host pointer mode.
  ///
  /// @return InvalidConfig for another device; or the GpuRuntime error of a
  ///         launch or a gemv.
  [[nodiscard]] Status rdm1(DeviceResources &res, const double *c, double *gamma);

  /// @brief The number of determinants, alpha.size() · beta.size(): the
  /// operator's n.
  [[nodiscard]] int32_t size() const noexcept { return n_alpha_ * n_beta_; }

private:
  CiSigma(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
          int32_t alpha_per, int32_t beta_per);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  int32_t beta_per_;
  bool has_integrals_ = false;
  double e_core_ = 0.0;
  // The link tables, three int32_t per replacement: target, packed pair, sign.
  Buffers::device<int32_t> alpha_links_;
  Buffers::device<int32_t> beta_links_;
  Buffers::device<uint64_t> alpha_strings_;
  Buffers::device<uint64_t> beta_strings_;
  // n_pairs × (n_pairs + 1), column-major: column pair(t, u) is ½ (tu|vw) over
  // the rows pair(v, w), and the last column k_vw.
  Buffers::device<double> contraction_;
  // n + 2 n²: h_ii, (ii|jj) and (ij|ji), for the diagonal.
  Buffers::device<double> diagonal_terms_;
};

} // namespace free_breakfast
