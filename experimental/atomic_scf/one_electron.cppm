/// @file
/// @brief One atom's one-electron matrices over the Cartesian layout: S, T and
/// the attraction V to its own nucleus.

// An internal partition: nothing here is exported. atomic_integrals calls
// these and carries their results to the spherical layout.
module free_breakfast.experimental.atomic_scf:one_electron;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;

namespace free_breakfast {

/// @brief S over @p basis's Cartesian layout, nao_cart x nao_cart,
/// column-major. Every shell of @p basis is taken to sit on one center.
[[nodiscard]] std::vector<double> one_center_overlap(const BasisSet &basis);

/// @brief T = -1/2 <a|nabla^2|b>, as @ref one_center_overlap.
[[nodiscard]] std::vector<double> one_center_kinetic(const BasisSet &basis);

/// @brief V, the attraction to a nucleus of charge @p z at the shells' common
/// center, as @ref one_center_overlap.
[[nodiscard]] std::vector<double> one_center_nuclear(int32_t z, const BasisSet &basis);

} // namespace free_breakfast
