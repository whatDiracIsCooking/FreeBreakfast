/// @file
/// @brief One atom's two-electron integrals over the Cartesian layout.

// An internal partition: nothing here is exported. atomic_integrals calls it
// and carries the tensor to the spherical layout.
module free_breakfast.experimental.atomic_scf:eri;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;

namespace free_breakfast {

/// @brief (ab|cd) over @p basis's Cartesian layout, every shell on one center:
/// nao_cart^4 values, (ab|cd) at a + n*(b + n*(c + n*d)).
///
/// (ab|cd) = 2 pi^{5/2} / (p q sqrt(p+q)) Sum_{tuv} E^{ab}_{tuv}
///   Sum_{t'u'v'} (-1)^{t'+u'+v'} E^{cd}_{t'u'v'} R_{t+t',u+u',v+v'}(alpha),
/// alpha = p q / (p + q), over every shell quartet with no permutational
/// symmetry: an atom's quartet count is small.
[[nodiscard]] std::vector<double> one_center_eri(const BasisSet &basis);

} // namespace free_breakfast
