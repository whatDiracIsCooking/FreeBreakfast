/// @file
/// @brief The Superposition-of-Atomic-Densities (SAD) initial guess: each
/// atom's spherically averaged atomic SCF density on its block of the
/// molecular density matrix, in the spherical AO layout (@ref
/// free_breakfast::sad_density).
///
/// **The model.** One restricted Hartree-Fock per atom, with fractional
/// occupations, over the atom's own shells. Its Z electrons (the neutral atom's
/// count) fill subshells in the Madelung order; the k-th subshell of momentum l
/// is seated on the k-th lowest orbital of the atom's l block, its electrons
/// spread evenly over the 2l+1 m. That density is spherical, so its Fock matrix
/// is block-diagonal by (l, m) and the same for every m: the SCF diagonalizes
/// one small radial block per l -- its shells of that momentum -- which is
/// PySCF's @c AtomSphAverageRHF exactly. The energy is the restricted
/// closed-shell expression over the spin-summed density, F = H + J - K/2, so a
/// fractionally occupied orbital keeps part of its own self-interaction.
///
/// **One center, so no Boys function.** Every integral of the atomic SCF has all
/// four centers at the nucleus: the Hermite expansion reduces to its
/// same-center recurrence and the Boys function is only ever evaluated at 0.
/// The integrals are exact, closed form, on the host.
///
/// **What the trace means.** Each atomic block holds Z electrons against the
/// atom's overlap, so trace(D S) is Sum_A Z_A exactly; trace(D) is not.
///
/// **Neutral atoms.** The guess places Sum_A Z_A electrons whatever the
/// molecule's net charge; rescaling or re-placing charge is the SCF driver's
/// call.
///
/// **Cartesian consumers.** The guess is spherical because every shell in this
/// tree is. A Cartesian-layout density is @c U D @c U^T with
/// @c U = spherical_transform_matrix(basis) -- density transforms contravariantly,
/// the opposite way to the @c U^T M U used for Fock and overlap matrices.
///
/// The atomic SCF is the internal partitions under @c atomic_scf/; nothing of
/// it is exported.
export module free_breakfast.sad;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The SAD initial-guess density: each atom's spherically averaged
/// atomic SCF density on its block of @p basis's spherical layout, zeros
/// between atoms.
///
/// One SCF per distinct (Z, shells) pair, so a molecule pays for its elements,
/// not its atoms: @ref place gives every atom of an element the same shells.
///
/// @param basis A basis made by @ref place over @p atoms, so it carries the
///              per-atom shell grouping (@ref BasisSet::natoms).
/// @param atoms The atoms, in the order @p basis was placed over: a
///              @c Molecule's @c atoms(). A span rather than the @c Molecule,
///              whose nuclei would need a device; the guess is host-only.
/// @return n x n, n = @c basis.nao_sphe(), flat column-major, element (i, j) at
///   index i + j*n. ErrorKind::InvalidConfig if @c basis.natoms() differs from
///   the atom count -- including a basis built from a bare shell list, which
///   carries no grouping at all -- or if some atom's basis has fewer linearly
///   independent orbitals of some momentum than it has occupied subshells of
///   it. ErrorKind::Unsupported for an atom past 96 Cartesian AOs, where its
///   n^4 two-electron tensor no longer fits in well under a gigabyte.
///   ErrorKind::MaxIterationsExceeded if an atomic SCF does not converge.
[[nodiscard]] Result<std::vector<double>> sad_density(const BasisSet &basis,
                                                      std::span<const Atom> atoms);

} // namespace free_breakfast
