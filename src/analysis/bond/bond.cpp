/// @file
/// @brief bond_orbitals: each orbital's weights summed per atom and per
/// angular momentum, the atoms above the threshold kept, and the orbital
/// pi when every one of them is almost all p.

module free_breakfast.analysis.bond;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

namespace free_breakfast {

Result<std::vector<BondOrbital>> bond_orbitals(const BasisSet &minimal, const int32_t n_orb,
                                               const std::span<const double> weights,
                                               const double threshold, const double pi_threshold) {
  const uint32_t n_atoms = minimal.natoms();
  if (n_atoms == 0) {
    return err_config("bond_orbitals: the minimal basis has no atoms");
  }
  const size_t m = minimal.nao_sphe();
  if (n_orb < 0 || weights.size() != m * static_cast<size_t>(n_orb)) {
    return err_config("bond_orbitals: {} weights are not {} x {}", weights.size(), m, n_orb);
  }
  if (!(threshold > 0.0 && threshold <= 1.0)) {
    return err_config("bond_orbitals: threshold {} is outside (0, 1]", threshold);
  }
  if (!(pi_threshold > 0.0 && pi_threshold <= 1.0)) {
    return err_config("bond_orbitals: pi_threshold {} is outside (0, 1]", pi_threshold);
  }
  constexpr size_t kMomenta = std::tuple_size_v<decltype(BondCenter::character)>;
  // Each function's atom and angular momentum.
  std::vector<uint32_t> owner(m);
  std::vector<size_t> momentum(m);
  for (uint32_t atom = 0; atom < n_atoms; ++atom) {
    const uint32_t first = minimal.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + minimal.atom_shell_count(atom); ++shell) {
      const int32_t l = minimal.shells()[shell].angular_momentum();
      if (std::cmp_greater_equal(l, kMomenta)) {
        return err_config("bond_orbitals: atom {} has an l = {} shell; s through f are read", atom,
                          l);
      }
      const uint32_t ao = minimal.ao_offset_sphe(shell);
      std::fill_n(owner.begin() + ao, (2 * l) + 1, atom);
      std::fill_n(momentum.begin() + ao, (2 * l) + 1, static_cast<size_t>(l));
    }
  }

  std::vector<BondOrbital> orbitals(static_cast<size_t>(n_orb));
  std::vector<BondCenter> atoms(n_atoms);
  for (size_t i = 0; i < orbitals.size(); ++i) {
    for (uint32_t a = 0; a < n_atoms; ++a) {
      atoms[a] = BondCenter{.atom = a};
    }
    for (size_t r = 0; r < m; ++r) {
      BondCenter &center = atoms[owner[r]];
      const double w = weights[r + (m * i)];
      center.population += w;
      center.character[momentum[r]] += w;
    }
    // Stable, so equal populations keep the molecule's order.
    std::ranges::stable_sort(atoms, std::greater{}, &BondCenter::population);
    std::vector<BondCenter> &centers = orbitals[i].centers;
    for (const BondCenter &center : atoms) {
      if (!centers.empty() && center.population < threshold) {
        break;
      }
      centers.push_back(center);
      if (center.population > 0.0) {
        for (double &share : centers.back().character) {
          share /= center.population;
        }
      }
    }
    const bool pi = std::ranges::all_of(
        centers, [&](const BondCenter &center) { return center.character[1] >= pi_threshold; });
    orbitals[i].kind = pi ? BondKind::Pi : BondKind::Sigma;
  }
  return orbitals;
}

} // namespace free_breakfast
