/// @file
/// @brief The vector pieces optimize and optimize_meci both step with: a
/// projection off an orthonormal basis, the max and rms norms, and the
/// Cholesky solve of the quasi-Newton step.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it. See docs/style.md, "Module-private code".

module free_breakfast.geometry:quasi_newton;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

namespace free_breakfast::geometry {

// v less its components along the orthonormal basis, in place.
void project_out(const std::vector<std::vector<double>> &basis, std::span<double> v) {
  for (const std::vector<double> &u : basis) {
    const double overlap = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    for (size_t j = 0; j < v.size(); ++j) {
      v[j] -= overlap * u[j];
    }
  }
}

// The largest |component| and the root mean square of v.
std::pair<double, double> max_and_rms(std::span<const double> v) {
  double largest = 0.0;
  double squares = 0.0;
  for (const double value : v) {
    largest = std::max(largest, std::abs(value));
    squares += value * value;
  }
  return {largest, std::sqrt(squares / static_cast<double>(v.size()))};
}

// p = -B^-1 g, by a Cholesky factorization of the n x n symmetric B
// (row-major). B is the BFGS Hessian, kept positive definite by updating it
// only over a positive curvature, so a pivot at or below zero means rounding
// has broken that and is reported, naming caller, rather than stepped over.
Result<std::vector<double>> newton_step(std::string_view caller, std::span<const double> hessian,
                                        std::span<const double> g) {
  const size_t n = g.size();
  std::vector<double> l(hessian.begin(), hessian.end());
  for (size_t j = 0; j < n; ++j) {
    double pivot = l[j * n + j];
    for (size_t k = 0; k < j; ++k) {
      pivot -= l[j * n + k] * l[j * n + k];
    }
    if (!(pivot > 0.0)) {
      return err_numerical("{}: the BFGS Hessian lost positive definiteness at pivot {} of {}",
                           caller, j, n);
    }
    l[j * n + j] = std::sqrt(pivot);
    for (size_t i = j + 1; i < n; ++i) {
      double value = l[i * n + j];
      for (size_t k = 0; k < j; ++k) {
        value -= l[i * n + k] * l[j * n + k];
      }
      l[i * n + j] = value / l[j * n + j];
    }
  }
  // L z = -g, then L^T p = z.
  std::vector<double> p(n);
  for (size_t i = 0; i < n; ++i) {
    double value = -g[i];
    for (size_t k = 0; k < i; ++k) {
      value -= l[i * n + k] * p[k];
    }
    p[i] = value / l[i * n + i];
  }
  for (size_t i = n; i-- > 0;) {
    double value = p[i];
    for (size_t k = i + 1; k < n; ++k) {
      value -= l[k * n + i] * p[k];
    }
    p[i] = value / l[i * n + i];
  }
  return p;
}

} // namespace free_breakfast::geometry
