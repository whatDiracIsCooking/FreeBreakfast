/// @file
/// @brief The same-center Hermite expansion and the zero-distance Hermite
/// Coulomb integrals.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;
import :hermite;

namespace free_breakfast {

std::vector<std::vector<double>> one_center_hermite(int32_t nmax, double p) {
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

double one_center_hermite_coulomb(int32_t t, int32_t u, int32_t v, double alpha) {
  if ((t % 2) != 0 || (u % 2) != 0 || (v % 2) != 0) {
    return 0.0;
  }
  const int32_t half = (t + u + v) / 2;
  return math::double_factorial_odd(t - 1) * math::double_factorial_odd(u - 1) *
         math::double_factorial_odd(v - 1) * std::pow(-2.0 * alpha, half) /
         static_cast<double>((2 * half) + 1);
}

} // namespace free_breakfast
