/// @file
/// @brief The one-center McMurchie-Davidson building blocks both integral
/// families share: the same-center Hermite expansion and the zero-distance
/// Hermite Coulomb integrals.

// A private interface partition: nothing here is exported. :one_electron and
// :eri build every integral from these two.
export module free_breakfast.experimental.atomic_scf:hermite;

import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;

namespace free_breakfast {

/// @brief Hermite expansion coefficients of x^n exp(-p x^2) about its own
/// center, for every n <= @p nmax: x^n exp(-p x^2) = Sum_t e[n][t] (d/dx)^t
/// exp(-p x^2).
///
/// With every McMurchie-Davidson distance zero, the two-center recurrence
/// reduces to e[n+1][t] = e[n][t-1] / (2p) + (t+1) e[n][t+1], e[0][0] = 1, and
/// e[n][t] vanishes unless t and n have the same parity.
[[nodiscard]] std::vector<std::vector<double>> one_center_hermite(int32_t nmax, double p) {
  const auto size = static_cast<size_t>(nmax) + 1;
  // One spare column, so e[n][t + 1] is in range at t = n + 1.
  std::vector<std::vector<double>> e(size, std::vector<double>(size + 1, 0.0));
  e[0][0] = 1.0;
  for (size_t n = 0; n + 1 < size; ++n) {
    for (size_t t = 0; t <= n + 1; ++t) {
      const double down = (t > 0) ? e[n][t - 1] / (2.0 * p) : 0.0;
      const double up = static_cast<double>(t + 1) * e[n][t + 1];
      e[n + 1][t] = down + up;
    }
  }
  return e;
}

/// @brief R_{tuv}(alpha) at zero distance: the (t, u, v)-th derivative of
/// F_0(alpha R^2) at R = 0.
///
/// Only even t, u, v survive, and then it is
///   (t-1)!! (u-1)!! (v-1)!! (-2 alpha)^{N/2} / (N + 1),   N = t + u + v,
/// the recurrence R^n_{t+1} = t R^{n+1}_{t-1} unrolled down to
/// R^n_{000} = (-2 alpha)^n F_n(0) with F_n(0) = 1 / (2n + 1).
[[nodiscard]] double one_center_hermite_coulomb(int32_t t, int32_t u, int32_t v, double alpha) {
  if ((t % 2) != 0 || (u % 2) != 0 || (v % 2) != 0) {
    return 0.0;
  }
  const int32_t half = (t + u + v) / 2;
  return math::double_factorial_odd(t - 1) * math::double_factorial_odd(u - 1) *
         math::double_factorial_odd(v - 1) * std::pow(-2.0 * alpha, half) /
         static_cast<double>((2 * half) + 1);
}

} // namespace free_breakfast
