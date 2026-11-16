/// @file
/// @brief Reading of an XYZ geometry into its atoms.
///
/// A parser returning through the project's Result<T> so a malformed file
/// becomes a classified Error rather than a crash or a thrown exception
/// escaping to the caller.
///
/// XYZ coordinates are conventionally in Angstrom, while an Atom's position is
/// in bohr (atomic units), so every coordinate is multiplied by
/// bohr_per_angstrom on read. That conversion is the one physical assumption
/// baked into this module. Host-only: a caller that wants the nuclei on a
/// device builds `Molecule(res, atoms, charge, multiplicity)` itself.

export module free_breakfast.io.xyz:reader;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.atom;

export namespace free_breakfast {

/// @brief Parse an XYZ geometry from an in-memory document into its atoms.
///
/// The standard XYZ layout is honoured: line 1 is the atom count, line 2 is a
/// free-form comment, and each subsequent line is `symbol x y z` with the
/// coordinates in Angstrom. The atoms come back in file order, their positions
/// in bohr. A count of zero, a count that disagrees with the number of atom
/// lines, an unknown element symbol, or a non-numeric coordinate each yields
/// ErrorKind::IO.
[[nodiscard]] Result<std::vector<Atom>> parse_xyz(std::string_view xyz_text);

/// @brief Parse an XYZ geometry from a file on disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_xyz.
[[nodiscard]] Result<std::vector<Atom>> read_xyz(const std::filesystem::path &path);

} // namespace free_breakfast
