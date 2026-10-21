/// @file
/// @brief One atom's SCF: @ref free_breakfast::AtomicScf and @ref
/// free_breakfast::atomic_scf.
export module free_breakfast.experimental.atomic_scf:scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

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

} // namespace free_breakfast
