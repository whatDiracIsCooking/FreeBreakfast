/// @file
/// @brief The molecular SAD guess from atomic SCFs: @ref
/// free_breakfast::atomic_scf_density.
export module free_breakfast.experimental.atomic_scf:density;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The SAD initial guess from atomic SCFs: each atom's @ref
/// AtomicScf::density on its block of @p basis's spherical layout, zeros
/// between atoms.
///
/// One SCF per distinct (Z, shells) pair, so a molecule pays for its elements,
/// not its atoms: @ref place gives every atom of an element the same shells.
///
/// @param basis A basis made by @ref place over @p atoms, so it carries the
///              per-atom shell grouping (@ref BasisSet::natoms).
/// @param atoms The atoms, in the order @p basis was placed over.
/// @return nao_sphe x nao_sphe, column-major. ErrorKind::InvalidConfig if
///   @c basis.natoms() differs from the atom count; otherwise the first
///   atom's @ref atomic_scf failure.
[[nodiscard]] Result<std::vector<double>> atomic_scf_density(const BasisSet &basis,
                                                             std::span<const Atom> atoms);

} // namespace free_breakfast
