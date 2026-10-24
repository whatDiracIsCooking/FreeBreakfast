/// @file
/// @brief The spin-summed reduced density matrices of CI vectors: the 1-RDM of
/// one vector, and the state-averaged 1- and 2-RDM of several, on the device
/// and on the host.
///
/// The host 2-RDM is written for clarity rather than speed: it is the
/// reference the device one is tested against.

export module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.ci.strings;

export namespace free_breakfast {

/// @brief γ_tu = ⟨c| E_tu |c⟩ = Σ_σ ⟨c| a†_tσ a_uσ |c⟩ on the host, for a CI
/// vector over two StringSets in host_sigma's layout: PySCF's `make_rdm1`.
///
/// One pass over each determinant's two replacement lists, O(dim · N (n − N +
/// 1)). Its trace is N_α + N_β for a unit @p c.
///
/// @param c The CI vector, alpha.size() · beta.size().
/// @param gamma Written with γ, n × n row-major.
/// @return InvalidConfig when the sets span different orbital counts or a span
///         has the wrong size; nothing is written then.
[[nodiscard]] Status rdm1(const StringSet &alpha, const StringSet &beta, std::span<const double> c,
                          std::span<double> gamma);

/// @brief The state-averaged spin-summed 1- and 2-RDM of CI vectors on the
/// device: γ = Σ_r w_r γ^(r), and
/// Γ_tuvw = Σ_r w_r ⟨c_r| E_tu E_vw − δ_uv E_tw |c_r⟩, PySCF's `make_rdm12`
/// (Γ_pqrs = ⟨a†_p a†_r a_s a_q⟩, spin summed). A single state is
/// weights = {1}.
///
/// create uploads the two sets' replacement lists, once per (n, N_α, N_β), so
/// a CASSCF's every macro-iteration reuses them. rdm12 then gathers, per root,
/// the one-replacement intermediate D_{tu,K} = ⟨c| E_tu |K⟩ over each
/// determinant's own lists, one thread per determinant, in chunks of
/// determinants; M += w_r Dᵀ D is a syrk and γ += w_r Dᵀ c a gemm per chunk,
/// and a kernel writes Γ_tuvw = M_{tu,wv} − δ_uv γ_tw, since ⟨K| E_vw |c⟩ =
/// D_{wv,K}. Nothing is atomic, so the result does not depend on the run.
/// Cost is O(Σ_r dim · n⁴) in the syrks.
///
/// Move-only; the buffers are on the device of the DeviceResources create was
/// given, which every call must pass again.
class CiRdm {
public:
  /// @brief Upload @p alpha's and @p beta's replacement lists to @p res's
  /// device. A failed allocation aborts (CreatePolicy).
  ///
  /// @return InvalidConfig when the two sets span different orbital counts or
  ///         none, or alpha.size() · beta.size() does not fit an int32_t; or
  ///         the GpuRuntime error of an upload.
  [[nodiscard]] static Result<CiRdm> create(DeviceResources &res, const StringSet &alpha,
                                            const StringSet &beta);

  CiRdm(CiRdm &&) noexcept = default;
  CiRdm &operator=(CiRdm &&) noexcept = default;
  CiRdm(const CiRdm &) = delete;
  CiRdm &operator=(const CiRdm &) = delete;
  ~CiRdm() = default;

  /// @brief γ and Γ of the @p weights.size() CI vectors @p vectors at
  /// @p weights, all on the device. M and one 64 MiB chunk of D are
  /// res.device_workspace(). Queued on res.stream() with res.blas(), whose
  /// pointer mode must be host (its default); it does not synchronize.
  ///
  /// @param weights w_r on the host, one per vector: each ≥ 0, summing to 1
  ///        within 1e-12.
  /// @param vectors Device, size() × weights.size(), root r at
  ///        [r · size(), (r + 1) · size()) in host_sigma's layout: what
  ///        CiRoots::vectors holds. Unit vectors, for γ and Γ to be RDMs.
  /// @param gamma Device, written with γ, n × n row-major.
  /// @param gamma2 Device, written with Γ, n⁴ row-major (t, u, v, w).
  /// @return InvalidConfig, with nothing queued, for another device, no
  ///         weights, a negative (or NaN) weight, or weights not summing to 1;
  ///         or the GpuRuntime error of a launch, the syrk or the gemm.
  [[nodiscard]] Status rdm12(DeviceResources &res, std::span<const double> weights,
                             const double *vectors, double *gamma, double *gamma2);

  /// @brief The number of determinants, alpha.size() · beta.size().
  [[nodiscard]] int32_t size() const noexcept { return n_alpha_ * n_beta_; }

private:
  CiRdm(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
        int32_t alpha_per, int32_t beta_per);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  int32_t beta_per_;
  // Three int32_t per replacement: target, unpacked pair p · n + q, sign.
  Buffers::device<int32_t> alpha_links_;
  Buffers::device<int32_t> beta_links_;
};

/// @brief CiRdm::rdm12's γ and Γ on the host, written for clarity: its
/// oracle.
///
/// γ is Σ_r w_r rdm1(c_r). For Γ, per root and determinant K, d_tu =
/// ⟨c| E_tu |K⟩ over K's lists, and Γ_tuvw += w_r d_tu d_wv (the second factor
/// ⟨K| E_vw |c⟩); then Γ_tuvw −= δ_uv γ_tw. Cost is O(Σ_r dim · n⁴).
///
/// @return InvalidConfig, with nothing written, when the sets span different
///         orbital counts, @p vectors is not weights.size() · dim long, the
///         weights are refused as CiRdm::rdm12 refuses them, or @p gamma or
///         @p gamma2 has the wrong size.
[[nodiscard]] Status host_rdm12(const StringSet &alpha, const StringSet &beta,
                                std::span<const double> vectors, std::span<const double> weights,
                                std::span<double> gamma, std::span<double> gamma2);

} // namespace free_breakfast
