/// @file
/// @brief S² on a CI vector, over the replacement lists.

module free_breakfast.ci.solver;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.ci.strings;

namespace free_breakfast {

Status host_spin_square(const StringSet &alpha, const StringSet &beta,
                        const std::span<const double> c, const std::span<double> s2c) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (alpha.n_orbitals() != beta.n_orbitals() || c.size() != dim || s2c.size() != dim) {
    return err_config("ci spin_square: orbitals {} and {}, {} determinants, given c of {} and "
                      "s2c of {}",
                      alpha.n_orbitals(), beta.n_orbitals(), dim, c.size(), s2c.size());
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  const auto nb = static_cast<size_t>(beta.size());

  // S² = S_z (S_z + 1) + N_β − Σ_pq E^β_pq E^α_qp. The constant part first.
  const double two_sz = alpha.n_electrons() - beta.n_electrons();
  const double constant = 0.25 * two_sz * (two_sz + 2.0) + beta.n_electrons();
  for (size_t i = 0; i < dim; ++i) {
    s2c[i] = constant * c[i];
  }

  // Then, per determinant K, scatter −<J|E^β_pq E^α_qp|K> c_K. An α entry of
  // K_α's list is E^α_ab (a created, b annihilated); its partner in K_β's
  // list is E^β_ba, found through a table of that list by (p, q), since each
  // (p, q) appears at most once. Both operators are even in their spin, so
  // each carries only its own list's sign.
  std::vector<int32_t> slot(n * n);
  for (int32_t kb = 0; kb < beta.size(); ++kb) {
    const std::span<const Replacement> beta_list = beta.replacements(kb);
    std::ranges::fill(slot, -1);
    for (size_t j = 0; j < beta_list.size(); ++j) {
      slot[static_cast<size_t>(beta_list[j].p) * n + static_cast<size_t>(beta_list[j].q)] =
          static_cast<int32_t>(j);
    }
    for (int32_t ka = 0; ka < alpha.size(); ++ka) {
      const double ck = c[static_cast<size_t>(ka) * nb + static_cast<size_t>(kb)];
      if (ck == 0.0) {
        continue;
      }
      for (const Replacement &ra : alpha.replacements(ka)) {
        const int32_t j = slot[static_cast<size_t>(ra.q) * n + static_cast<size_t>(ra.p)];
        if (j < 0) {
          continue;
        }
        const Replacement &rb = beta_list[static_cast<size_t>(j)];
        s2c[static_cast<size_t>(ra.target) * nb + static_cast<size_t>(rb.target)] -=
            ra.sign * rb.sign * ck;
      }
    }
  }
  return {};
}

} // namespace free_breakfast
