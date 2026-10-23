/// @file
/// @brief Parsing of a Basis Set Exchange (BSE) JSON basis definition into a
/// geometry-free @ref free_breakfast::BasisStamp, and placing it onto a
/// molecule's atoms.
///
/// The Basis Set Exchange (https://www.basissetexchange.org) publishes basis
/// sets as a JSON document. The shape this module consumes is:
///
/// @code
/// {
///   "elements": {
///     "1":  { "electron_shells": [ ... ] },     // keyed by atomic number (Z)
///     "8":  { "electron_shells": [ ... ] }       // as a decimal string
///   }
/// }
/// @endcode
///
/// Each entry of @c electron_shells is a contracted-shell block with
///
///   * @c angular_momentum -- a list of ints. Usually one (a pure s/p/d shell),
///     but an sp-shell carries `[0, 1]`.
///   * @c exponents -- a list of primitive Gaussian exponents, written as
///     decimal STRINGS.
///   * @c coefficients -- a list of contraction columns, each a list of decimal
///     STRINGS matched one-to-one to the exponents. A general contraction has
///     several columns for one angular momentum; an sp-shell has one column per
///     angular momentum.
///   * @c function_type -- OPTIONAL. BSE's own enum is
///     `["gto", "gto_spherical", "gto_cartesian", "sto"]`; there is no separate
///     harmonic-type field, this is the only place BSE says whether a shell
///     means Cartesian or real-spherical functions.
///     **Only spherical input is accepted**, because every @ref
///     free_breakfast::ShellStamp is spherical. See @ref check_spherical for
///     the exact table -- in short, `gto_spherical` is accepted,
///     `gto_cartesian` is rejected, bare `gto` (or the field's absence) is
///     spherical for `angular_momentum <= 1` (the two conventions coincide
///     there) and rejected as ambiguous otherwise, and `sto` (or anything
///     else) is rejected outright. @ref free_breakfast::spherical_transform_matrix
///     bridges the Cartesian functions the integral kernels evaluate onto the
///     spherical layout.
///
/// This module turns that document into a geometry-free @ref
/// free_breakfast::BasisStamp -- a per-element table of centerless shell
/// templates -- which @ref free_breakfast::place then stamps onto a @ref
/// free_breakfast::Molecule: for every atom, in molecule order, the element's
/// shells are emitted at that atom's center.
///
/// A block with @c M angular momenta and @c N coefficient columns is mapped to
/// shells by pairing momentum @c i with column @c i (the sp / spd convention),
/// so `angular_momentum` and `coefficients` must have the same length. The one
/// exception is a single angular momentum with several columns (a general
/// contraction), where each column becomes its own shell of that momentum.
/// @c function_type is a property of the whole block, so every shell it emits
/// resolves the SAME @c function_type against its OWN angular momentum.
///
/// Everything stays in atomic units: exponents are in inverse bohr^2 and centers
/// in bohr, exactly as @ref free_breakfast::Shell expects, so no
/// unit conversion happens here.
///
/// Every failure -- malformed JSON, a missing field, a wrong JSON type, a
/// non-numeric string, an element the molecule needs but the basis lacks --
/// comes back through @ref Result as a classified @ref Error rather than a throw
/// or a crash.

export module free_breakfast.io.basis:reader;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.molecule;

export namespace free_breakfast {

/// @brief Parse a BSE JSON basis document into a geometry-free @ref
/// BasisStamp.
///
/// The result is the element -> shell-stamps table a basis-set file specifies:
/// no positions. @ref place stamps it onto a molecule's atoms. Splitting parse
/// from place lets a single parsed basis be laid over many geometries with the
/// per-primitive normalization (@ref BasisStamp::primitive_normalized)
/// computed once, since it is center-independent.
///
/// Syntactically invalid JSON yields ErrorKind::IO. A document that parses but
/// is missing @c elements or a shell field, or has one of the wrong JSON type,
/// or carries a non-numeric exponent/coefficient string, yields
/// ErrorKind::InvalidConfig.
[[nodiscard]] Result<BasisStamp> parse_basis(std::string_view json_text);

/// @brief Parse a BSE JSON basis from a file on disk into a @ref BasisStamp.
///
/// A file that cannot be opened yields ErrorKind::IO; otherwise the contents
/// flow through @ref parse_basis and carry its classification.
[[nodiscard]] Result<BasisStamp> read_basis(const std::filesystem::path &path);

/// @brief Parse a BSE JSON basis and @ref place it onto a molecule in one step.
///
/// Convenience over @ref parse_basis followed by @ref place; the classification
/// of whichever step fails is preserved.
[[nodiscard]] Result<BasisSet> parse_and_assign_basis(std::string_view json_text,
                                                      const Molecule &molecule);

/// @brief Parse a density-fitting (auxiliary) BSE JSON basis and @ref place it
/// onto the SAME molecule the orbital basis was built over.
///
/// Density fitting needs a second @ref BasisSet -- the fitting set (e.g.
/// @c cc-pVDZ-RI, @c def2-universal-JKFIT) -- laid over the identical geometry
/// as the orbital basis. There is no separate parse: a fitting set is an
/// ordinary BSE record, so this reuses @ref parse_basis and @ref place verbatim.
/// The distinct name exists to make the DF call site read for what it is, to
/// carry the L-ceiling caveat below, and to give the pairing its own error text.
///
/// **Pairing.** The aux and orbital sets are paired by being @ref place'd onto
/// one @ref Molecule: same atom count, same centers, same bohr units, by
/// construction rather than by a separate cross-check -- the shell centers of
/// both sets come from that molecule's atoms. The one way the pairing can fail
/// is coverage: a fitting record that lacks an element the molecule uses cannot
/// be placed. That comes back as ErrorKind::InvalidConfig (from @ref place),
/// naming the missing element -- an aux set that cannot cover the molecule is a
/// configuration error, not a silently smaller basis.
///
/// **Angular-momentum ceiling.** Fitting sets routinely run to @c f and @c g,
/// above the s/p/d ceiling (@ref kMaxAngularMomentum) the GPU integral kernels
/// are designed for. Parsing and HOLDING those higher-L shells is fully
/// supported here -- @ref ShellStamp and @ref BasisSet impose no cap, exactly as
/// for an orbital basis. Feeding such a basis to the GPU ERI path is a SEPARATE
/// concern the kernels reject on their own; this reader neither enforces nor
/// silently drops the ceiling.
[[nodiscard]] Result<BasisSet> parse_and_assign_aux_basis(std::string_view json_text,
                                                          const Molecule &molecule);

/// @brief Whether @p spec names a basis set compiled into this binary.
///
/// Spelling is folded before the comparison -- lowercased, with @c -, @c _ and
/// spaces dropped -- so @c STO-3G, @c sto-3g and @c sto3g are one name.
///
/// This is the question a front end has to ask BEFORE it resolves a relative
/// path, since a name must not be joined to a config file's directory. @ref
/// resolve_basis asks it again and needs no help: a caller that simply wants
/// the basis should call that, not this.
[[nodiscard]] bool is_builtin_basis(std::string_view spec);

/// @brief The basis @p spec asks for, by compiled-in name or by file path.
///
/// A spec that names one of the compiled-in sets (@ref is_builtin_basis) is
/// parsed from the bytes in the binary; anything else is read from disk as an
/// ordinary BSE JSON file, which is how a basis this build does not carry is
/// used without rebuilding. The two cannot be told apart by shape, so the name
/// wins: a file called @c sto-3g in the working directory is reached as
/// @c ./sto-3g.
///
/// Classification is @ref read_basis's throughout: a spec that is neither a
/// known name nor an openable file is ErrorKind::IO, as an unreadable path is
/// everywhere else in this layer, with the compiled-in names appended to the
/// message since a misspelt name is the likelier mistake.
[[nodiscard]] Result<BasisStamp> resolve_basis(std::string_view spec);

} // namespace free_breakfast
