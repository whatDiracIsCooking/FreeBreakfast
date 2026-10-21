/// @file
/// @brief One atom's integrals over the spherical layout of its shells: @ref
/// free_breakfast::AtomicIntegrals and @ref free_breakfast::atomic_integrals.
export module free_breakfast.experimental.atomic_scf:integrals;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief One atom's integrals over the spherical layout of its shells:
/// @c n = Sum of each shell's @c function_count_sphe(), every matrix flat and
/// column-major.
struct AtomicIntegrals {
  uint32_t nao_sphe{0};        ///< n, the spherical AO count.
  std::vector<double> overlap; ///< S, n x n.
  std::vector<double> kinetic; ///< T, n x n.
  std::vector<double> nuclear; ///< V, n x n: attraction to the atom's own nucleus, charge z.
  std::vector<double> eri;     ///< (ij|kl) at i + n*(j + n*(k + n*l)), n^4.
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

} // namespace free_breakfast
