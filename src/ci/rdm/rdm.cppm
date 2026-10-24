/// @file
/// @brief The spin-summed 1-RDM of a CI vector, on the host.

export module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
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

} // namespace free_breakfast
