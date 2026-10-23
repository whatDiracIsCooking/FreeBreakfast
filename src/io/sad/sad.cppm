/// @file
/// @brief The compiled-in SAD atomic densities: every `data/sad/<basis>/<Z>.npy`,
/// found by basis-set name and atomic number (@ref
/// free_breakfast::builtin_sad_density).
///
/// A domain module over the format module @c free_breakfast.io.npy, as
/// @c free_breakfast.io.basis is over @c free_breakfast.io.json. Its one
/// function is a @c SadStampLookup: @c free_breakfast.sad takes it as a
/// parameter rather than importing this module, so the generator that writes
/// the files (@c tools/sad) can link the SCF without linking its own output.

export module free_breakfast.io.sad;

import std;
import free_breakfast.core.types;

export namespace free_breakfast {

/// @brief The baked SAD density of element @p z in the compiled-in basis set
/// @p basis, or nullopt if this binary carries none.
///
/// The block is the atom's converged spherically averaged density over its own
/// spherical AOs, n x n flat column-major, n the atom's spherical AO count:
/// what @c sad_density's SCF computes for that atom, written ahead of time by
/// @c tools/sad.
///
/// @param basis The set's canonical name exactly as @c BasisSet::name holds it
///              (`cc-pVDZ`), not a spelling to fold: only a basis resolved from
///              the binary carries a name, and it carries the canonical one.
/// @param z     The atomic number.
///
/// Safe to call concurrently: it reads constant data and parses a copy. The
/// bytes are this build's own, checked at configure time and parsed by the
/// tests, so bytes that do not parse as a square matrix abort.
[[nodiscard]] std::optional<std::vector<double>> builtin_sad_density(std::string_view basis,
                                                                     int32_t z);

} // namespace free_breakfast
