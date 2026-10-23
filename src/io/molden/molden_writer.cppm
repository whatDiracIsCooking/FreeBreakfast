/// @file
/// @brief The write half of the Molden module: atoms, basis and orbitals ->
/// Molden text.
///
/// The orbitals are over @ref BasisSet's spherical layout, which the file
/// declares with @c [5D7F] and @c [9G]. The format has no flag above g, so a
/// shell above l = 4 is refused rather than written as something no reader
/// would take for what it is.

export module free_breakfast.io.molden:writer;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.atom;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief Serialize orbitals to Molden text.
///
/// @param atoms       The nuclei, in the order @p basis was placed over them
///                    (@c Molecule::atoms).
/// @param basis       A basis made by @ref place: it carries the per-atom
///                    shell grouping @c [GTO] is written by, and its
///                    coefficients have the primitive normalization folded in
///                    (@c BasisStamp::primitive_normalized), the convention
///                    every integral in this tree assumes.
/// @param c           The MO coefficients, @c nao_sphe x @c nmo, column-major,
///                    one column per orbital, rows in @p basis's spherical
///                    layout (@c m = -l..+l per shell).
/// @param energies    One orbital energy per column of @p c (hartree).
/// @param occupations One occupation per column of @p c.
///
/// Each shell's components are written in Molden's m = 0, +1, -1, +2, -2, ...
/// order, and each row of @p c is rescaled from this tree's AO to the
/// unit-normalized contraction every Molden reader builds from @c [GTO]. Every
/// orbital is written as Alpha, with symmetry label A.
///
/// Sizes that disagree, or a basis whose per-atom grouping does not cover
/// @p atoms, are ErrorKind::InvalidConfig; a shell above g is
/// ErrorKind::Unsupported.
[[nodiscard]] Result<std::string> molden_to_string(std::span<const Atom> atoms,
                                                   const BasisSet &basis, std::span<const double> c,
                                                   std::span<const double> energies,
                                                   std::span<const double> occupations);

/// @brief Write orbitals to @p path as a Molden file; see @ref molden_to_string.
[[nodiscard]] Status write_molden(const std::filesystem::path &path, std::span<const Atom> atoms,
                                  const BasisSet &basis, std::span<const double> c,
                                  std::span<const double> energies,
                                  std::span<const double> occupations);

} // namespace free_breakfast
