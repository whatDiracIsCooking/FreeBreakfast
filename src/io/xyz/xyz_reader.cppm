/// @file
/// @brief Reading of an XYZ geometry into a @ref free_breakfast::Molecule.
///
/// A parser returning through the project's Result<T> so a malformed file
/// becomes a classified Error rather than a crash or a thrown exception
/// escaping to the caller.
///
/// XYZ coordinates are conventionally in Angstrom, while Molecule stores
/// positions in bohr (atomic units), so every coordinate is multiplied by
/// bohr_per_angstrom on read. That conversion is the one physical assumption
/// baked into this module.

export module free_breakfast.io.xyz:reader;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.molecule;

export namespace free_breakfast {

/// @brief Parse an XYZ geometry from an in-memory document into a Molecule.
///
/// The standard XYZ layout is honoured: line 1 is the atom count, line 2 is a
/// free-form comment, and each subsequent line is `symbol x y z` with the
/// coordinates in Angstrom. Coordinates are converted to bohr on the way into
/// the Molecule. A count that disagrees with the number of atom lines, an
/// unknown element symbol, or a non-numeric coordinate each yields
/// ErrorKind::IO.
///
/// @param xyz_text       The XYZ document.
/// @param charge         Net molecular charge to attach (XYZ carries none).
/// @param multiplicity   Spin multiplicity to attach (XYZ carries none).
[[nodiscard]] Result<Molecule> parse_xyz(std::string_view xyz_text, std::int32_t charge = 0,
                                         std::int32_t multiplicity = 1);

/// @brief Parse an XYZ geometry from a file on disk.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through parse_xyz.
[[nodiscard]] Result<Molecule> read_xyz(const std::filesystem::path &path, std::int32_t charge = 0,
                                        std::int32_t multiplicity = 1);

} // namespace free_breakfast
