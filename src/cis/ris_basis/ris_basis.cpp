/// @file
/// @brief The Ghosh radius table and the RIS shells placed from it.

module free_breakfast.cis.ris_basis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kBohrPerAngstrom
import free_breakfast.core.error_handling;
import free_breakfast.atom;
import free_breakfast.molecule; // nuclear_charge
import free_breakfast.basis;

namespace free_breakfast::cis {

// The Ghosh (GB) absolute atomic radii in Angstrom, element Z at index Z - 1:
// gpu4pyscf's tdscf/parameter.py `radii`, verbatim, for H..Kr. The README has
// where they come from and how they were checked.
constexpr std::array<double, 36> kGhoshRadii{
    0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618, // H..Ne
    2.165,  1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056,                 // Na..Ar
    3.293,  2.5419, 2.4149, 2.2998, 2.1953, 2.1,    2.0124, 1.9319, 1.8575,         // K..Co
    1.7888, 1.725,  1.6654, 1.4489, 1.2823, 1.145,  1.0424, 0.9532, 0.8782,         // Ni..Kr
};

Result<BasisSet> ris_basis(const std::span<const Atom> atoms, const double theta,
                           const RisShells shells) {
  if (!std::isfinite(theta) || theta <= 0.0) {
    return err_config("cis::ris_basis: theta must be finite and positive, got {}", theta);
  }
  const int32_t heavy_max_l = shells == RisShells::S ? 0 : shells == RisShells::Sp ? 1 : 2;

  std::vector<Shell> placed;
  std::vector<uint32_t> atom_shell_offsets{0};
  atom_shell_offsets.reserve(atoms.size() + 1);
  for (const Atom &atom : atoms) {
    const int32_t z = nuclear_charge(atom);
    if (z < 1 || z > static_cast<int32_t>(kGhoshRadii.size())) {
      return err_config("cis::ris_basis: no Ghosh radius for Z={}; the table covers 1..{}", z,
                        kGhoshRadii.size());
    }
    const double radius = kGhoshRadii[static_cast<size_t>(z - 1)] * kBohrPerAngstrom<double>;
    const double exponent = theta / (radius * radius);
    const int32_t max_l = z == 1 ? 0 : heavy_max_l;
    for (int32_t l = 0; l <= max_l; ++l) {
      const ShellStamp stamp{l, {Primitive{exponent, 1.0}}};
      placed.push_back(place(stamp.primitive_normalized(), atom.position));
    }
    atom_shell_offsets.push_back(static_cast<uint32_t>(placed.size()));
  }
  return BasisSet{std::move(placed), std::move(atom_shell_offsets)};
}

} // namespace free_breakfast::cis
