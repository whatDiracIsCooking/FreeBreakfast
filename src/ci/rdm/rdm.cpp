/// @file
/// @brief The spin-summed 1-RDM.

module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.ci.strings;

namespace free_breakfast {

Status rdm1(const StringSet &alpha, const StringSet &beta, const std::span<const double> c,
            const std::span<double> gamma) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  if (alpha.n_orbitals() != beta.n_orbitals() || c.size() != dim || gamma.size() != n * n) {
    return err_config("ci rdm1: orbitals {} and {}, {} determinants, given c of {} and gamma of {}",
                      alpha.n_orbitals(), beta.n_orbitals(), dim, c.size(), gamma.size());
  }
  const auto nb = static_cast<size_t>(beta.size());
  std::ranges::fill(gamma, 0.0);
  // a†_p a_q |K> = sign |J> adds c_J sign c_K to γ_pq, for each spin's list.
  for (int32_t ka = 0; ka < alpha.size(); ++ka) {
    for (int32_t kb = 0; kb < beta.size(); ++kb) {
      const double ck = c[static_cast<size_t>(ka) * nb + static_cast<size_t>(kb)];
      for (const Replacement &r : alpha.replacements(ka)) {
        const size_t j = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
      for (const Replacement &r : beta.replacements(kb)) {
        const size_t j = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
    }
  }
  return {};
}

} // namespace free_breakfast
