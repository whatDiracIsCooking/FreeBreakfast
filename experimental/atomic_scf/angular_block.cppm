/// @file
/// @brief One momentum's radial block of the atomic SCF.

// An internal partition: nothing here is exported.
module free_breakfast.experimental.atomic_scf:angular_block;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :integrals;

namespace free_breakfast {

/// @brief One momentum's radial block: the atom's shells of that l, in order,
/// the orthonormal span they cover, and the subshells the aufbau fill seats on
/// its orbitals.
///
/// A spherical density makes the Fock matrix block-diagonal by (l, m) and the
/// same for every m, so each block's matrices are n x n over its shells -- the
/// m-average of the full spherical matrix -- not (n (2l+1))^2.
struct AngularBlock {
  int32_t l{0};                   ///< The momentum.
  std::vector<size_t> ao_first;   ///< Each shell's first spherical AO, its m = -l.
  size_t n{0};                    ///< Shells of this l.
  std::vector<double> overlap;    ///< n x n.
  std::vector<double> x;          ///< n x m canonical orthogonalizer, X^T S X = 1.
  size_t m{0};                    ///< Linearly independent orbitals.
  std::vector<double> occupation; ///< Per orbital, spin-summed, over all 2l+1 m.
  std::vector<double> density;    ///< n x n, per m: Sum_k occ_k / (2l+1) c_k c_k^T.

  /// @brief The m-averaged block of the full spherical @p nao x @p nao matrix.
  [[nodiscard]] std::vector<double> average(const std::vector<double> &full, size_t nao) const;

  /// @brief Diagonalize @p fock (n x n) in the block's orthonormal span and
  /// refresh @ref density from the result.
  /// @return The orbital energies, ascending, m of them.
  std::vector<double> diagonalize(const std::vector<double> &fock);
};

/// @brief One block per l from 0 to the highest momentum in @p shells, each
/// with its overlap, its orthogonalizer and its share of @p z's aufbau fill.
/// Densities are left empty, for the caller's first @ref
/// AngularBlock::diagonalize.
///
/// @return ErrorKind::InvalidConfig if some l has more occupied subshells than
///   independent orbitals to seat them.
[[nodiscard]] Result<std::vector<AngularBlock>>
angular_blocks(int32_t z, std::span<const Shell> shells, const AtomicIntegrals &ints);

/// @brief The full spherical @p nao x @p nao density: each block's
/// @ref AngularBlock::density on every one of its m.
[[nodiscard]] std::vector<double> angular_density(const std::vector<AngularBlock> &blocks,
                                                  size_t nao);

} // namespace free_breakfast
