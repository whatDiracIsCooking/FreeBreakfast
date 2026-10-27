/// @file
/// @brief The host references the CI tests check the device against: σ = Hc,
/// diag(H), the spin-summed 1- and 2-RDMs and S² c, each a plain loop over the
/// strings' replacement lists.
///
/// Written for clarity rather than speed, and linked by the test executables
/// alone: no calculation runs them. Every vector is in ci::Layout::full, element
/// I_α · beta.size() + I_β, which is `fci.direct_spin1`'s layout.

export module free_breakfast.test.ci_oracles;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.ci.strings;

export namespace free_breakfast::test {

/// @brief σ = H c on the host, H = E_core + Σ_tu h_tu E_tu
/// + ½ Σ_tuvw (tu|vw) (E_tu E_vw − δ_uv E_tw), E_tu the spin-summed
/// excitation operator: ci::Sigma::apply's reference.
///
/// The α–α, β–β and α–β terms all go through the two sets' replacement lists.
/// Cost is O(dim · n⁴), memory O(n²) besides the vectors.
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
[[nodiscard]] Status host_sigma(const ci::StringSet &alpha, const ci::StringSet &beta,
                                double e_core, std::span<const double> h_eff,
                                std::span<const double> eri, std::span<const double> c,
                                std::span<double> sigma);

/// @brief diag(H) on the host, for the same H and layout as host_sigma:
/// ci::Sigma::diagonal's reference.
///
/// Slater–Condon's rule for a determinant's diagonal over its occupied spin
/// orbitals, α (ascending) then β (ascending), i < j:
/// E_core + Σ_i h_ii + Σ_{i<j} [(ii|jj) − δ_{σ_i σ_j} (ij|ji)], summed in that
/// order, which is ci::Sigma::diagonal's.
///
/// @param diagonal Written with H_II, alpha.size() · beta.size().
/// @return InvalidConfig as host_sigma does.
[[nodiscard]] Status host_diagonal(const ci::StringSet &alpha, const ci::StringSet &beta,
                                   double e_core, std::span<const double> h_eff,
                                   std::span<const double> eri, std::span<double> diagonal);

/// @brief γ_tu = ⟨c| E_tu |c⟩ = Σ_σ ⟨c| a†_tσ a_uσ |c⟩ on the host, PySCF's
/// `make_rdm1`: ci::Sigma::rdm1's reference.
///
/// One pass over each determinant's two replacement lists, O(dim · N (n − N +
/// 1)). Its trace is N_α + N_β for a unit @p c.
///
/// @param c The CI vector, alpha.size() · beta.size().
/// @param gamma Written with γ, n × n row-major.
/// @return InvalidConfig when the sets span different orbital counts or a span
///         has the wrong size; nothing is written then.
[[nodiscard]] Status host_rdm1(const ci::StringSet &alpha, const ci::StringSet &beta,
                               std::span<const double> c, std::span<double> gamma);

/// @brief ci::Rdm::rdm12's γ and Γ on the host: its reference.
///
/// γ is Σ_r w_r host_rdm1(c_r). For Γ, per root and determinant K, d_tu =
/// ⟨c| E_tu |K⟩ over K's lists, and Γ_tuvw += w_r d_tu d_wv (the second factor
/// ⟨K| E_vw |c⟩); then Γ_tuvw −= δ_uv γ_tw. Cost is O(Σ_r dim · n⁴).
///
/// @return InvalidConfig, with nothing written, when the sets span different
///         orbital counts, @p vectors is not weights.size() · dim long, the
///         weights are refused as ci::Rdm::rdm12 refuses them, or @p gamma or
///         @p gamma2 has the wrong size.
[[nodiscard]] Status host_rdm12(const ci::StringSet &alpha, const ci::StringSet &beta,
                                std::span<const double> vectors, std::span<const double> weights,
                                std::span<double> gamma, std::span<double> gamma2);

/// @brief S² c on the host: ci::SpinSquare::apply's reference.
///
/// S² = S_z (S_z + 1) + S₋S₊, and S₋S₊ = N_β − Σ_pq E^β_pq E^α_qp, so both
/// terms go through the sets' own replacement lists: nothing leaves the
/// (N_α, N_β) sector. Cost is O(dim · N_α (n − N_α + 1)).
///
/// @param c The CI vector, alpha.size() · beta.size().
/// @param s2c Written with S² c, the same size; must not overlap @p c.
/// @return InvalidConfig when the sets span different orbital counts or a span
///         has the wrong size; nothing is written then.
[[nodiscard]] Status host_spin_square(const ci::StringSet &alpha, const ci::StringSet &beta,
                                      std::span<const double> c, std::span<double> s2c);

} // namespace free_breakfast::test
