/// @file
/// @brief The RIS minimal auxiliary basis (Zhou & Parker 2024; gpu4pyscf
/// @c tdscf/ris.py): one uncontracted shell per emitted l on each atom, of
/// exponent θ / R_A², R_A the atom's Ghosh (GB) radius in bohr.
///
/// The README has the radii's provenance and how they were checked.

export module free_breakfast.cis.ris_basis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.atom;
import free_breakfast.basis;

export namespace free_breakfast::cis {

/// @brief The shells a heavy atom (Z > 1) gets; hydrogen gets the s shell
/// alone under all three. gpu4pyscf's @c fitting_basis strings 's', 'sp' and
/// 'spd'.
enum class RisShells : uint8_t {
  S,
  Sp,
  Spd,
};

/// @brief The RIS auxiliary basis over @p atoms, in atom order: per atom one
/// single-primitive shell per l that @p shells emits, all of exponent
/// θ / R_A², R_A the Ghosh radius in bohr.
///
/// The same kind of @ref BasisSet @c pipeline.eri's density fitting places:
/// spherical, its coefficients primitive-normalized (@ref
/// ShellStamp::primitive_normalized), and carrying the per-atom shell map
/// (@ref BasisSet::natoms). It has no @ref BasisSet::name. Pass
/// @c molecule.atoms().
///
/// @param theta  The exponent scale θ; gpu4pyscf's default is 0.2.
/// @param shells What a heavy atom gets; gpu4pyscf's J default is Sp.
/// @return InvalidConfig unless @p theta is finite and positive and every
///         atom's Z is in 1..36, the extent of the radius table.
[[nodiscard]] Result<BasisSet> ris_basis(std::span<const Atom> atoms, double theta = 0.2,
                                         RisShells shells = RisShells::Sp);

} // namespace free_breakfast::cis
