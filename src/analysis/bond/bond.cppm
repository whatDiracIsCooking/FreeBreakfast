/// @file
/// @brief Bond-orbital analysis: each orbital read as a core or lone pair, a
/// bond, or a multi-center bond, by the atoms it sits on and the angular
/// character of its share of each (Knizia, J. Chem. Theory Comput. 9, 4834
/// (2013), section 3.2: "98% on two centers", "28% s-character").
///
/// Host arithmetic over the orbitals' weights on an orthonormal atom-owned
/// minimal basis -- orbital_weights over the IAOs, which over the IBOs is
/// the paper's bonding picture. The README has the rules.

export module free_breakfast.analysis.bond;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief One atom an orbital sits on.
struct BondCenter {
  /// @brief The atom, in the molecule's order.
  uint32_t atom = 0;
  /// @brief The orbital's population on the atom, a fraction of its norm.
  double population = 0.0;
  /// @brief The s, p, d and f shares of @ref population, summing to one.
  std::array<double, 4> character{};
};

/// @brief Whether an orbital is a sigma or a pi orbital.
enum class BondKind : uint8_t {
  Sigma, ///< Some center's share is not almost entirely p.
  Pi,    ///< Every center's share is almost entirely p.
};

/// @brief One orbital's centers: one for a core or lone pair, two for a
/// bond, three or more for a multi-center bond.
struct BondOrbital {
  /// @brief The atoms holding at least the threshold of the orbital, by
  /// descending population; at least the largest one.
  std::vector<BondCenter> centers;
  /// @brief Pi when every center's p character is at least the pi
  /// threshold, sigma otherwise; one-center orbitals included.
  BondKind kind = BondKind::Sigma;
};

/// @brief The centers of each of @p n_orb orbitals, from their @p weights on
/// @p minimal's functions.
///
/// @p weights is m x @p n_orb, column-major, m = minimal.nao_sphe(), orbital
/// i's weight on function r at [r + m * i]: what orbital_weights returns. An
/// atom is a center of an orbital when its population there, the weights
/// summed over the atom's functions, is at least @p threshold; the largest
/// atom is a center regardless, so no orbital has none. The orbital is pi
/// when each center's p share, character[1], is at least @p pi_threshold,
/// and sigma otherwise (Derricotte and Evangelista, J. Chem. Theory Comput.
/// 13, 5984 (2017), section 2.2), so a pure-p lone pair is pi too.
///
/// @return One BondOrbital per orbital; or InvalidConfig when @p minimal has
///         no atoms or a shell above f, @p weights is not m x @p n_orb, or
///         @p threshold or @p pi_threshold is outside (0, 1].
[[nodiscard]] Result<std::vector<BondOrbital>> bond_orbitals(const BasisSet &minimal, int32_t n_orb,
                                                             std::span<const double> weights,
                                                             double threshold = 0.1,
                                                             double pi_threshold = 0.9);

} // namespace free_breakfast
