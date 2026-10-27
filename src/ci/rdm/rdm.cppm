/// @file
/// @brief The state-averaged spin-summed 1- and 2-RDM of CI vectors, on the
/// device. The 1-RDM of one vector is CiSigma::rdm1.
///
/// The host reference it is tested against is test/ci_oracles' host_rdm12.

export module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma; // CiLayout

export namespace free_breakfast {

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
/// Over one StringSet for both spins, rdm12 also takes swap-symmetric vectors
/// in CiLayout::symmetric. E_tu commutes with the swap of the two strings, so
/// D_K is D of K's mirror, and the sums over every K are sums over the
/// triangle with weight 2 off the diagonal: the gather takes the triangle's
/// rows alone, each off-diagonal one scaled by √2, and the packed c carries
/// the other √2 of γ's product. That halves the gather and the syrk.
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
  /// @param vectors Device, size(@p layout) × weights.size(), root r at
  ///        [r · size(layout), (r + 1) · size(layout)): in CiLayout::full,
  ///        what CiRoots::vectors holds, or packed in CiLayout::symmetric.
  ///        Unit vectors, for γ and Γ to be RDMs.
  /// @param gamma Device, written with γ, n × n row-major.
  /// @param gamma2 Device, written with Γ, n⁴ row-major (t, u, v, w).
  /// @param layout The vectors' layout; CiLayout::symmetric needs one set for
  ///        both spins.
  /// @return InvalidConfig, with nothing queued, for another device, no
  ///         weights, a negative (or NaN) weight, weights not summing to 1, or
  ///         CiLayout::symmetric over two different sets; or the GpuRuntime
  ///         error of a launch, the syrk or the gemm.
  [[nodiscard]] Status rdm12(DeviceResources &res, std::span<const double> weights,
                             const double *vectors, double *gamma, double *gamma2,
                             CiLayout layout = CiLayout::full);

  /// @brief The length of a vector: the number of determinants,
  /// alpha.size() · beta.size(), or for CiLayout::symmetric the triangle's,
  /// alpha.size() (alpha.size() + 1) / 2.
  [[nodiscard]] int32_t size(CiLayout layout = CiLayout::full) const noexcept {
    return layout == CiLayout::symmetric ? n_alpha_ * (n_alpha_ + 1) / 2 : n_alpha_ * n_beta_;
  }

private:
  CiRdm(DeviceResources &res, int32_t n_orbitals, int32_t n_alpha, int32_t n_beta,
        int32_t alpha_per, int32_t beta_per, bool one_set);

  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  int32_t beta_per_;
  bool one_set_; // alpha and beta are the same (n, N): CiLayout::symmetric applies
  // Three int32_t per replacement: target, unpacked pair p · n + q, sign.
  Buffers::device<int32_t> alpha_links_;
  Buffers::device<int32_t> beta_links_;
};

} // namespace free_breakfast
