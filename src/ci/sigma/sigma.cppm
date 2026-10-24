/// @file
/// @brief σ = Hc and diag(H) on the host, for a determinant CI vector over
/// two StringSets, from the active-space integrals as plain arrays.
///
/// Written for clarity rather than speed: it is the reference the GPU sigma
/// is tested against.

export module free_breakfast.ci.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
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

} // namespace free_breakfast
