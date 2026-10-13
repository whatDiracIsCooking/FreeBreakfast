/// @file
/// @brief Small exact-in-double combinatorics shared by the basis partitions.
///
/// A private interface partition: it exports nothing, so no importer of
/// free_breakfast.basis can name these, but the inline functions of
/// @c :harmonics and @c :basis_set that call them still reach their
/// definitions. See docs/style.md, "Module-private code".

export module free_breakfast.basis:numeric;

namespace free_breakfast {

// n!! over odd n (n * (n-2) * ... * 1); n <= 1 is the empty product, 1.0.
// :harmonics uses it for the Gaussian-moment inner product, :basis_set for the
// per-shell sqrt((2l-1)!!) spherical scale. ShellStamp::primitive_normalized
// computes the same quantity inline to fold it into a primitive's coefficient;
// :basis_set rescales by the reciprocal amount, which is why the two agree.
[[nodiscard]] inline double double_factorial_odd(int n) {
  double result = 1.0;
  for (int k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

// n! as a double, exact for the small n a basis's l ever reaches.
[[nodiscard]] inline double factorial(int n) {
  double result = 1.0;
  for (int k = 2; k <= n; ++k) {
    result *= static_cast<double>(k);
  }
  return result;
}

// Binomial coefficient C(n, k) as a double, exact for the small n this ever
// sees (n <= l; a basis's l is tiny). Zero outside 0 <= k <= n.
[[nodiscard]] inline double binomial(int n, int k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  double result = 1.0;
  for (int i = 0; i < k; ++i) {
    result = result * static_cast<double>(n - i) / static_cast<double>(i + 1);
  }
  return result;
}

} // namespace free_breakfast
