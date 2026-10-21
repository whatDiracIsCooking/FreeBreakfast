/// @file
/// @brief The spherically averaged atomic SCF a production SAD guess is made
/// of: one atom's restricted Hartree-Fock with fractional occupations, on the
/// host, over its own one-center integrals (@ref free_breakfast::atomic_scf),
/// and the block-diagonal molecular guess assembled from it (@ref
/// free_breakfast::atomic_scf_density).
///
/// Experimental: built only under FREE_BREAKFAST_BUILD_EXPERIMENTAL. The
/// README has the model and how it relates to free_breakfast.sad.
///
/// **The model.** free_breakfast.sad's occupations, with orbitals in place of
/// basis functions. The atom's Z electrons fill subshells in @ref kAufbauOrder;
/// the k-th subshell of momentum l is seated on the k-th lowest orbital of the
/// l block, its electrons spread evenly over the 2l+1 m. That density is
/// spherical, so its Fock matrix is block-diagonal by (l, m) and the same for
/// every m: the SCF diagonalizes one small radial block per l -- its shells of
/// that momentum -- which is PySCF's @c AtomSphAverageRHF exactly. The energy
/// is the restricted closed-shell expression over the spin-summed density,
/// E = 1/2 tr D (H + F) with F = H + J - K/2, so a fractionally occupied
/// orbital keeps part of its own self-interaction (hydrogen is not -0.5).
///
/// **One center, so no Boys function.** Every integral here has all four
/// centers at the nucleus. In McMurchie-Davidson terms every distance is zero,
/// the Hermite expansion of a product reduces to its same-center recurrence,
/// and the Boys function is only ever evaluated at 0, where F_n(0) = 1/(2n+1).
/// The integrals are exact, closed form, with no quadrature or table.
export module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief One atom's integrals over the spherical layout of its shells:
/// @c n = Sum of each shell's @c function_count_sphe(), every matrix flat and
/// column-major.
struct AtomicIntegrals {
  uint32_t nao_sphe{0};          ///< n, the spherical AO count.
  std::vector<double> overlap;   ///< S, n x n.
  std::vector<double> kinetic;   ///< T, n x n.
  std::vector<double> nuclear;   ///< V, n x n: attraction to the atom's own nucleus, charge z.
  std::vector<double> repulsion; ///< (ij|kl) at i + n*(j + n*(k + n*l)), n^4.
};

/// @brief S, T, V and the two-electron integrals of one atom's @p shells, over
/// their spherical layout.
///
/// Computed over the Cartesian components and brought to the spherical layout
/// with @ref spherical_transform_matrix of the shells, so the convention is the
/// tree's: coefficients multiply unnormalized Cartesian primitives, and the
/// spherical columns run m = -l..+l.
///
/// @param z      The nuclear charge V attracts to.
/// @param shells The atom's shells, all on one center (which one does not
///               matter: the integrals are translation invariant).
/// @return ErrorKind::InvalidConfig if @p shells is empty or its centers
///   differ; ErrorKind::Unsupported past 96 Cartesian AOs, where the n^4
///   two-electron tensor no longer fits in well under a gigabyte.
[[nodiscard]] Result<AtomicIntegrals> atomic_integrals(int32_t z, std::span<const Shell> shells);

/// @brief A converged atomic SCF.
struct AtomicScf {
  double energy{0.0};    ///< Total electronic energy, hartree (an atom has no nuclear repulsion).
  int32_t iterations{0}; ///< Fock builds it took.
  /// Spin-summed density over the atom's spherical layout, n x n,
  /// column-major: its trace against @ref AtomicIntegrals::overlap is z.
  std::vector<double> density;
  /// Per l, from 0 to the highest momentum of the shells, the l block's orbital
  /// energies ascending (one per m-degenerate level; empty for a momentum with
  /// no shell).
  std::vector<std::vector<double>> orbital_energies;
};

/// @brief The spherically averaged, fractionally occupied RHF of a neutral atom
/// of atomic number @p z in @p shells.
///
/// Starts from the core Hamiltonian, accelerates with Pulay DIIS on each l
/// block's commutator F P S - S P F, and stops when the largest commutator
/// element and the energy change both fall under @p tolerance.
///
/// @param z              Atomic number in [0, 118] (outside aborts, as for
///                       @ref subshell_occupation); 0 gives an empty density.
/// @param shells         The atom's shells, as for @ref atomic_integrals.
/// @param tolerance      On the commutator and the energy change, hartree.
/// @param max_iterations Fock builds before giving up.
/// @return ErrorKind::InvalidConfig, as @ref atomic_integrals, or if some l
///   has more occupied subshells than linearly independent orbitals to seat
///   them (free_breakfast.sad drops those electrons; an SCF over the rest would
///   be of a different ion). ErrorKind::Unsupported as @ref atomic_integrals.
///   ErrorKind::MaxIterationsExceeded if it has not converged by then.
[[nodiscard]] Result<AtomicScf> atomic_scf(int32_t z, std::span<const Shell> shells,
                                           double tolerance = 1e-9, int32_t max_iterations = 128);

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
