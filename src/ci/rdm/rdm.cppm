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

/// @brief The state-averaged spin-summed 1- and 2-RDM of @p weights.size() CI
/// vectors, on the device: γ = Σ_r w_r γ^(r), and
/// Γ_tuvw = Σ_r w_r ⟨c_r| E_tu E_vw − δ_uv E_tw |c_r⟩, PySCF's `make_rdm12`
/// (Γ_pqrs = ⟨a†_p a†_r a_s a_q⟩, spin summed). A single state is
/// weights = {1}.
///
/// Per root, the one-replacement intermediate D_{tu,K} = ⟨c| E_tu |K⟩ is
/// gathered over each determinant's own lists, one thread per determinant, in
/// chunks of determinants; then M += w_r D Dᵀ and γ += w_r D c are two gemms
/// per chunk, and Γ_tuvw = M_{tu,wv} − δ_uv γ_tw, since ⟨K| E_vw |c⟩ =
/// D_{wv,K}. The vectors, the link tables, M, γ and one chunk of D are
/// res.device_workspace(); nothing is atomic, so the result does not depend on
/// the run. Cost is O(Σ_r dim · n⁴) in the gemms. Queued on res.stream() with
/// res.blas(), whose pointer mode must be host (its default), and synchronized
/// before returning.
///
/// @param vectors The CI vectors on the host, root r at [r · dim, (r + 1) ·
///        dim) in host_sigma's layout, dim = alpha.size() · beta.size(): what
///        CiRoots::vectors holds. Unit vectors, for γ and Γ to be RDMs.
/// @param weights w_r, one per vector: each ≥ 0, summing to 1 within 1e-12.
/// @param gamma Written with γ, n × n row-major.
/// @param gamma2 Written with Γ, n⁴ row-major (t, u, v, w).
/// @return InvalidConfig, with nothing written, when the sets span different
///         orbital counts, dim does not fit an int32_t, @p vectors is not
///         weights.size() · dim long, a weight is negative (or NaN), the
///         weights do not sum to 1, or @p gamma or @p gamma2 has the wrong
///         size; or the GpuRuntime error of an upload, a launch or a gemm.
[[nodiscard]] Status rdm12(DeviceResources &res, const StringSet &alpha, const StringSet &beta,
                           std::span<const double> vectors, std::span<const double> weights,
                           std::span<double> gamma, std::span<double> gamma2);

/// @brief rdm12's γ and Γ on the host, written for clarity: its oracle.
///
/// γ is Σ_r w_r rdm1(c_r). For Γ, per root and determinant K, d_tu =
/// ⟨c| E_tu |K⟩ over K's lists, and Γ_tuvw += w_r d_tu d_wv (the second factor
/// ⟨K| E_vw |c⟩); then Γ_tuvw −= δ_uv γ_tw. Cost is O(Σ_r dim · n⁴).
///
/// @return InvalidConfig as rdm12 does (dim is not bounded here); nothing is
///         written then.
[[nodiscard]] Status host_rdm12(const StringSet &alpha, const StringSet &beta,
                                std::span<const double> vectors, std::span<const double> weights,
                                std::span<double> gamma, std::span<double> gamma2);

} // namespace free_breakfast
