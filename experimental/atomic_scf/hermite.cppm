/// @file
/// @brief The one-center McMurchie-Davidson building blocks both integral
/// families share: the same-center Hermite expansion and the zero-distance
/// Hermite Coulomb integrals.

// An internal partition: nothing here is exported. :one_electron and :eri
// build every integral from these two.
module free_breakfast.experimental.atomic_scf:hermite;

import std;
import free_breakfast.core.types;

namespace free_breakfast {

/// @brief Hermite expansion coefficients of x^n exp(-p x^2) about its own
/// center, for every n <= @p nmax: x^n exp(-p x^2) = Sum_t e[n][t] (d/dx)^t
/// exp(-p x^2).
///
/// With every McMurchie-Davidson distance zero, the two-center recurrence
/// reduces to e[n+1][t] = e[n][t-1] / (2p) + (t+1) e[n][t+1], e[0][0] = 1, and
/// e[n][t] vanishes unless t and n have the same parity.
[[nodiscard]] std::vector<std::vector<double>> one_center_hermite(int32_t nmax, double p);

/// @brief R_{tuv}(alpha) at zero distance: the (t, u, v)-th derivative of
/// F_0(alpha R^2) at R = 0.
///
/// Only even t, u, v survive, and then it is
///   (t-1)!! (u-1)!! (v-1)!! (-2 alpha)^{N/2} / (N + 1),   N = t + u + v,
/// the recurrence R^n_{t+1} = t R^{n+1}_{t-1} unrolled down to
/// R^n_{000} = (-2 alpha)^n F_n(0) with F_n(0) = 1 / (2n + 1).
[[nodiscard]] double one_center_hermite_coulomb(int32_t t, int32_t u, int32_t v, double alpha);

} // namespace free_breakfast
