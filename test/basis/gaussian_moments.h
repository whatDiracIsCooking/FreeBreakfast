/// @file
/// @brief The closed-form Gaussian-moment algebra the harmonics and basis-set
/// cases check against, written out independently of src/basis. Include it
/// after `import free_breakfast.basis;`.

#pragma once

namespace free_breakfast::test {

using Powers = std::array<std::int32_t, 3>;

// (n)!! over odd n, the empty product for n <= 1.
inline double double_factorial(std::int32_t n) {
  double result = 1.0;
  for (std::int32_t k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

// The angular Gaussian-moment inner product the harmonic table is normalized
// under: prod over axes of (p+q-1)!!, zero if any axis sum is odd.
inline double monomial_gram(const Powers &p, const Powers &q) {
  double result = 1.0;
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    const std::int32_t s = p[axis] + q[axis];
    if (s % 2 != 0) {
      return 0.0;
    }
    result *= double_factorial(s - 1);
  }
  return result;
}

// Row index of the Cartesian monomial `p` within its shell.
inline std::size_t row_of(std::int32_t l, const Powers &p) {
  const std::vector<Powers> powers = cartesian_powers(l);
  const auto it = std::ranges::find(powers, p);
  return static_cast<std::size_t>(it - powers.begin());
}

} // namespace free_breakfast::test
