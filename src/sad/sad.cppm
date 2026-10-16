/// @file
/// @brief The Superposition-of-Atomic-Densities (SAD) initial guess: per-atom
/// aufbau occupations laid onto the diagonal of the molecular density matrix,
/// in the spherical AO layout (@ref free_breakfast::sad_density).
///
/// **The model.** No atomic SCF is run. Each atom's Z electrons (the neutral
/// atom's count) fill subshells in the Madelung order (@ref kAufbauOrder); the
/// k-th subshell of momentum l is placed on the atom's k-th shell of momentum l,
/// and its electrons are spread evenly over that shell's 2l+1 spherical
/// functions. The real solid harmonics of one shell are mutually orthogonal and
/// share a norm, so an equal weight per m is a true spherical average -- which
/// it would not be over the Cartesian components, whose self-overlaps differ
/// (see @ref spherical_transform_matrix). Off-diagonal entries are zero.
///
/// **What the trace means.** Every electron placed lands on the diagonal, so
/// trace(D) is the placed electron count exactly. That is the orthonormal-AO
/// picture: shells of one atom overlap (1s with 2s), so the physical count
/// trace(D S) is close to, but not exactly, the same number.
///
/// **Neutral atoms.** The guess places Sum_A Z_A electrons whatever the
/// molecule's net charge; rescaling or re-placing charge is the SCF driver's
/// call.
///
/// **Cartesian consumers.** The guess is spherical because every shell in this
/// tree is. A Cartesian-layout density is @c U D @c U^T with
/// @c U = spherical_transform_matrix(basis) -- density transforms contravariantly,
/// the opposite way to the @c U^T M U used for Fock and overlap matrices.
export module free_breakfast.sad;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief One subshell in the aufbau filling order.
struct SubshellFill {
  int32_t l;        ///< Angular momentum (0=s, 1=p, 2=d, 3=f).
  int32_t capacity; ///< Electrons that fill it, 2(2l+1).
};

/// @brief The Madelung (n+l, then n) filling order, 1s 2s 2p 3s 3p 4s 3d 4p ...
/// 7p: long enough to seat any Z in [1, 118].
///
/// Written out rather than generated so the order is auditable at a glance.
inline constexpr std::array<SubshellFill, 19> kAufbauOrder{{
    {0, 2},  // 1s
    {0, 2},  // 2s
    {1, 6},  // 2p
    {0, 2},  // 3s
    {1, 6},  // 3p
    {0, 2},  // 4s
    {2, 10}, // 3d
    {1, 6},  // 4p
    {0, 2},  // 5s
    {2, 10}, // 4d
    {1, 6},  // 5p
    {0, 2},  // 6s
    {3, 14}, // 4f
    {2, 10}, // 5d
    {1, 6},  // 6p
    {0, 2},  // 7s
    {3, 14}, // 5f
    {2, 10}, // 6d
    {1, 6},  // 7p
}};

/// @brief Ground-state electron count per subshell for a neutral atom of atomic
/// number @p z, aligned to @ref kAufbauOrder.
///
/// A plain aufbau fill: full subshells hold their capacity, the last holds the
/// remainder, and trailing empty subshells are omitted, so the entries sum to
/// exactly @p z. The d/f anomalies (Cr, Cu, ...) are not encoded; a spherically
/// averaged guess does not need them.
///
/// @param z Atomic number in [0, 118]; 0 (@c Element::Sentinel) gives an empty
///   result. Anything outside that range aborts -- no @c Element spells it, so
///   it is a caller bug.
[[nodiscard]] std::vector<int32_t> subshell_occupation(int32_t z);

/// @brief The diagonal SAD occupations for one atom: one value per spherical AO
/// of @p shells, concatenated in shell order.
///
/// @param z      Atomic number of the atom, as for @ref subshell_occupation.
/// @param shells The shells the atom owns, in basis order.
/// @return Length Sum of each shell's @c function_count_sphe(). The values
///   sum to @p z when the basis has a shell for every occupied subshell; a
///   subshell with no shell left to sit on is dropped, and the sum falls short
///   by its electrons (an under-covered basis, not an error).
[[nodiscard]] std::vector<double> atomic_diagonal(int32_t z, std::span<const Shell> shells);

/// @brief The SAD guess's diagonal over @p basis's spherical layout: each atom's
/// @ref atomic_diagonal at that atom's AO range.
///
/// @param basis    A basis made by @ref place over @p molecule, so it carries
///                 the per-atom shell grouping (@ref BasisSet::natoms).
/// @param molecule The molecule whose atomic numbers key the occupations.
/// @return Length @c basis.nao_sphe(); its sum is the placed electron
///   count. ErrorKind::InvalidConfig if @c basis.natoms() differs from
///   @c molecule.natoms() -- including a basis built from a bare shell list,
///   which carries no grouping at all.
[[nodiscard]] Result<std::vector<double>> sad_diagonal(const BasisSet &basis,
                                                       const Molecule &molecule);

/// @brief The SAD initial-guess density: @ref sad_diagonal on the diagonal of
/// an n x n matrix, n = @c basis.nao_sphe(), zeros elsewhere.
///
/// Flat column-major, element (i, j) at index i + j*n. Fails exactly when @ref
/// sad_diagonal does.
[[nodiscard]] Result<std::vector<double>> sad_density(const BasisSet &basis,
                                                      const Molecule &molecule);

} // namespace free_breakfast
