/// @file
/// @brief The state-averaged spin-summed 1- and 2-RDM of CI vectors, and the
/// transition RDMs of two, on the device. The 1-RDM of one vector is
/// Sigma::rdm1.
///
/// The host reference it is tested against is test/ci_oracles' host_rdm12.

export module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma; // Layout

export namespace free_breakfast::ci {

/// @brief The state-averaged spin-summed 1- and 2-RDM of CI vectors on the
/// device: γ = Σ_r w_r γ^(r), and
/// Γ_tuvw = Σ_r w_r ⟨c_r| E_tu E_vw − δ_uv E_tw |c_r⟩, PySCF's `make_rdm12`
/// (Γ_pqrs = ⟨a†_p a†_r a_s a_q⟩, spin summed). A single state is
/// weights = {1}.
///
/// create borrows the two sets' replacement lists, already on the device in
/// their DeviceStringSets, which a CASSCF's every macro-iteration and its
/// sigma share. rdm12 then gathers, per root,
/// the one-replacement intermediate D_{tu,K} = ⟨c| E_tu |K⟩ over each
/// determinant's own lists, one thread per determinant, in chunks of
/// determinants; M += w_r Dᵀ D is a syrk and γ += w_r Dᵀ c a gemm per chunk,
/// and a kernel writes Γ_tuvw = M_{tu,wv} − δ_uv γ_tw, since ⟨K| E_vw |c⟩ =
/// D_{wv,K}. Nothing is atomic, so the result does not depend on the run.
/// Cost is O(Σ_r dim · n⁴) in the syrks.
///
/// Over one StringSet for both spins, rdm12 also takes swap-symmetric vectors
/// in Layout::symmetric. E_tu commutes with the swap of the two strings, so
/// D_K is D of K's mirror, and the sums over every K are sums over the
/// triangle with weight 2 off the diagonal: the gather takes the triangle's
/// rows alone, each off-diagonal one scaled by √2, and the packed c carries
/// the other √2 of γ's product. That halves the gather and the syrk.
///
/// Move-only; the tables are on the device of the DeviceResources create was
/// given, which every call must pass again, and @p alpha and @p beta must
/// outlive it.
class Rdm {
public:
  /// @brief An Rdm over @p alpha's and @p beta's device tables, which it
  /// reads in place (one set may be both).
  ///
  /// @return InvalidConfig when either set is on another device than @p res,
  ///         the two span different orbital counts or none, or
  ///         alpha.size() · beta.size() does not fit an int32_t.
  [[nodiscard]] static Result<Rdm> create(DeviceResources &res, const DeviceStringSet &alpha,
                                          const DeviceStringSet &beta);

  Rdm(Rdm &&) noexcept = default;
  Rdm &operator=(Rdm &&) noexcept = default;
  Rdm(const Rdm &) = delete;
  Rdm &operator=(const Rdm &) = delete;
  ~Rdm() = default;

  /// @brief γ and Γ of the @p weights.size() CI vectors @p vectors at
  /// @p weights, all on the device. M and one 64 MiB chunk of D are
  /// res.device_workspace(). Queued on res.stream() with res.blas(), whose
  /// pointer mode must be host (its default); it does not synchronize.
  ///
  /// @param weights w_r on the host, one per vector: each ≥ 0, summing to 1
  ///        within 1e-12.
  /// @param vectors Device, size(@p layout) × weights.size(), root r at
  ///        [r · size(layout), (r + 1) · size(layout)): in Layout::full,
  ///        what Roots::vectors holds, or packed in Layout::symmetric.
  ///        Unit vectors, for γ and Γ to be RDMs.
  /// @param gamma Device, written with γ, n × n row-major.
  /// @param gamma2 Device, written with Γ, n⁴ row-major (t, u, v, w).
  /// @param layout The vectors' layout; Layout::symmetric needs one set for
  ///        both spins.
  /// @return InvalidConfig, with nothing queued, for another device, no
  ///         weights, a negative (or NaN) weight, weights not summing to 1, or
  ///         Layout::symmetric over two different sets; or the GpuRuntime
  ///         error of a launch, the syrk or the gemm.
  [[nodiscard]] Status rdm12(DeviceResources &res, std::span<const double> weights,
                             const double *vectors, double *gamma, double *gamma2,
                             Layout layout = Layout::full);

  /// @brief The spin-summed transition 1-RDM γ^IJ_pq = ⟨bra| E_pq |ket⟩ =
  /// Σ_σ ⟨bra| a†_pσ a_qσ |ket⟩: the first index is the creation. Not
  /// symmetric (γ^JI = γ^IJᵀ), and the transpose of PySCF's
  /// `trans_rdm1(bra, ket)`, whose [p, q] is ⟨bra| a†_q a_p |ket⟩; so it is
  /// PySCF's `trans_rdm1(ket, bra)`. One chunk of D is
  /// res.device_workspace(); the gather is rdm12's over @p bra, and
  /// γ^IJ = D_braᵀ c_ket a gemm per chunk. Queued on res.stream() with
  /// res.blas() (host pointer mode); it does not synchronize.
  ///
  /// @param bra, ket Device, size(@p layout) each: one vector apiece, which
  ///        may be the same (γ^II is rdm12's γ at weight 1).
  /// @param gamma Device, written with γ^IJ, n × n row-major.
  /// @param layout The vectors' layout, both in it; Layout::symmetric needs
  ///        one set for both spins.
  /// @return InvalidConfig, with nothing queued, for another device or
  ///         Layout::symmetric over two different sets; or the GpuRuntime
  ///         error of a launch or the gemm.
  [[nodiscard]] Status trans_rdm1(DeviceResources &res, const double *bra, const double *ket,
                                  double *gamma, Layout layout = Layout::full);

  /// @brief The symmetrized spin-summed transition 1- and 2-RDMs,
  /// γ_pq = ½ (⟨bra| E_pq |ket⟩ + ⟨ket| E_pq |bra⟩) and
  /// Γ_tuvw = ½ (⟨bra| E_tu E_vw − δ_uv E_tw |ket⟩ + (bra ↔ ket)), in
  /// rdm12's (`make_rdm12`'s) convention: ½ (PySCF's
  /// `trans_rdm12(bra, ket)` + `trans_rdm12(ket, bra)`). Unlike PySCF's
  /// trans_rdm12 it is symmetric in bra and ket, γ is symmetric and
  /// Γ_tuvw = Γ_vwtu, as rdm12's are; bra = ket is rdm12 at weight 1, to
  /// rounding. The gather is rdm12's over both vectors, M =
  /// ½ (D_braᵀ D_ket + D_ketᵀ D_bra) a syr2k in place of the syrk and
  /// γ = ½ (D_braᵀ c_ket + D_ketᵀ c_bra) two gemms per chunk; M and two
  /// chunks of D, 64 MiB together, are res.device_workspace(). Queued on
  /// res.stream() with res.blas() (host pointer mode); it does not
  /// synchronize.
  ///
  /// @param bra, ket Device, size(@p layout) each, which may be the same.
  /// @param gamma Device, written with γ, n × n row-major.
  /// @param gamma2 Device, written with Γ, n⁴ row-major (t, u, v, w).
  /// @param layout As for trans_rdm1.
  /// @return As trans_rdm1's, or the GpuRuntime error of the syr2k.
  [[nodiscard]] Status trans_rdm12(DeviceResources &res, const double *bra, const double *ket,
                                   double *gamma, double *gamma2, Layout layout = Layout::full);

  /// @brief The length of a vector: the number of determinants,
  /// alpha.size() · beta.size(), or for Layout::symmetric the triangle's,
  /// alpha.size() (alpha.size() + 1) / 2.
  [[nodiscard]] int32_t size(Layout layout = Layout::full) const noexcept {
    return layout == Layout::symmetric ? n_alpha_ * (n_alpha_ + 1) / 2 : n_alpha_ * n_beta_;
  }

private:
  Rdm(DeviceResources &res, const DeviceStringSet &alpha, const DeviceStringSet &beta);

  // What every call refuses: another device, or Layout::symmetric over two
  // sets.
  [[nodiscard]] Status check_call(const DeviceResources &res, Layout layout) const;
  // D of @p c for the rows [@p begin, @p begin + @p count) into the zeroed
  // @p d (count × n² column-major), zeroing it first, on res.stream().
  [[nodiscard]] Status gather(DeviceResources &res, Layout layout, int32_t begin, int32_t count,
                              const double *c, double *d) const;
  int32_t dev_idx_;
  int32_t n_orbitals_;
  int32_t n_alpha_; // alpha.size()
  int32_t n_beta_;  // beta.size()
  int32_t alpha_per_;
  int32_t beta_per_;
  bool one_set_; // alpha and beta are the same (n, N): Layout::symmetric applies
  // The sets' device lists, borrowed: one per spin, the same for one set.
  const Replacement *alpha_links_;
  const Replacement *beta_links_;
};

} // namespace free_breakfast::ci
