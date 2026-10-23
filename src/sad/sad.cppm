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
/// **Baked stamps.** Each form takes an optional @ref SadStampLookup. For a
/// basis that carries a name (@ref BasisSet::name), it is asked for each
/// element's density first, and an element it has skips its SCF. This module
/// never knows where the densities come from: the compiled-in ones are
/// @c free_breakfast.io.sad's, and the generator that writes them links this
/// module and passes no lookup, so it can only ever compute.
///
/// The atomic SCF is the internal partitions under @c atomic_scf/; nothing of
/// it is exported.
export module free_breakfast.sad;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

// Re-exported: the task form of sad_density takes a tf::FlowBuilder and hands
// back a tf::Task, so a caller of it needs taskflow.
export import free_breakfast.third_party.taskflow;

export namespace free_breakfast {

/// @brief Where @ref sad_density finds an element's density instead of running
/// its SCF: given a basis set's name (@ref BasisSet::name, never empty) and an
/// atomic number, that atom's converged block over its own spherical AOs, flat
/// column-major, or nullopt for an element it does not have.
///
/// A plain function pointer, called from the task form's worker threads, so it
/// must be safe to call concurrently. @c free_breakfast.io.sad's
/// @c builtin_sad_density is the one in the tree.
using SadStampLookup = std::optional<std::vector<double>> (*)(std::string_view basis, int32_t z);

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
/// @param baked Asked first for each element's density when @p basis is
///              named; nullptr, or an unnamed basis, computes every one.
/// @return n x n, n = @c basis.nao_sphe(), flat column-major, element (i, j) at
///   index i + j*n. ErrorKind::InvalidConfig if @c basis.natoms() differs from
///   the atom count -- including a basis built from a bare shell list, which
///   carries no grouping at all -- or if some atom's basis has fewer linearly
///   independent orbitals of some momentum than it has occupied subshells of
///   it, or if @p baked returns a block of the wrong size for an atom's shells.
///   ErrorKind::Unsupported for an atom past 96 Cartesian AOs, where its
///   n^4 two-electron tensor no longer fits in well under a gigabyte.
///   ErrorKind::MaxIterationsExceeded if an atomic SCF does not converge.
[[nodiscard]] Result<std::vector<double>>
sad_density(const BasisSet &basis, std::span<const Atom> atoms, SadStampLookup baked = nullptr);

/// @brief The same guess as one task of @p flow, its atomic SCFs run in
/// parallel: when it runs, it spawns a subflow with one task per distinct
/// (Z, shells), each placing its stamp on its own atoms' blocks as soon as its
/// SCF converges.
///
/// The result is bitwise the serial form's, error included: each SCF is
/// independent of the others, and when several fail, @p out holds the error of
/// the first atom in order -- though, unlike the serial form, every other SCF
/// has still run.
///
/// **Lifetimes.** Nothing is read when this is called. @p basis, the atoms
/// @p atoms views and @p out are captured by reference and used when the task
/// runs, so they must outlive the run; @p basis may still be filled in by a
/// predecessor of the returned task. @p out is final once the task completes,
/// and is the only thing it writes.
///
/// @param flow  The graph to add the task to: a @c tf::Taskflow, or the
///              @c tf::Subflow of an enclosing task.
/// @param basis As for the serial form.
/// @param atoms As for the serial form.
/// @param out   Where the guess, or the serial form's error, is written.
/// @param baked As for the serial form; copied, so it need not outlive the
///              call.
/// @return The task, for the caller to order against the rest of @p flow.
tf::Task sad_density(tf::FlowBuilder &flow, const BasisSet &basis, std::span<const Atom> atoms,
                     Result<std::vector<double>> &out, SadStampLookup baked = nullptr);

} // namespace free_breakfast
