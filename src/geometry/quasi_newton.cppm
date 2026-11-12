/// @file
/// @brief The vector pieces optimize, optimize_meci and irc step with: a
/// projection off an orthonormal basis, the max and rms norms, the Cholesky
/// solve of the quasi-Newton step, and the symmetric eigensolve of irc's
/// hypersphere step.
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

// The eigenvalues of the n x n symmetric a (row-major), ascending, and the
// eigenvectors as the columns of the n x n row-major matrix returned beside
// them, by cyclic Jacobi rotations. Host-side because irc's matrices are
// 3N - 6 square, a few dozen at most, and its walk touches no device; Jacobi
// rather than a tridiagonal QL because it is short and its accuracy is
// relative to each eigenvalue, which the hypersphere's secular equation
// divides by. Sweeps stop when the off-diagonal is under 1e-30 of the
// Frobenius norm squared, or after 100 (never reached: Jacobi converges
// quadratically, in under 10 sweeps at these sizes).
std::pair<std::vector<double>, std::vector<double>> symmetric_eigen(std::vector<double> a,
                                                                    size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    v[i * n + i] = 1.0;
  }
  double total = 0.0;
  for (const double value : a) {
    total += value * value;
  }
  for (int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        off += a[p * n + q] * a[p * n + q];
      }
    }
    if (!(off > 1e-30 * total)) {
      break;
    }
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        const double apq = a[p * n + q];
        if (apq == 0.0) {
          continue;
        }
        // The rotation that zeroes a_pq: t = tan(theta), the smaller root of
        // t^2 + 2 t cot(2 theta) - 1 = 0, for stability.
        const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
        const double t =
            (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double c = 1.0 / std::sqrt((t * t) + 1.0);
        const double s = t * c;
        for (size_t k = 0; k < n; ++k) {
          const double akp = a[k * n + p];
          const double akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (size_t k = 0; k < n; ++k) {
          const double apk = a[p * n + k];
          const double aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (size_t k = 0; k < n; ++k) {
          const double vkp = v[k * n + p];
          const double vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  // Ascending, the columns of v permuted alike.
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::ranges::sort(order, {}, [&](size_t i) { return a[i * n + i]; });
  std::vector<double> values(n);
  std::vector<double> vectors(n * n);
  for (size_t j = 0; j < n; ++j) {
    values[j] = a[order[j] * n + order[j]];
    for (size_t i = 0; i < n; ++i) {
      vectors[i * n + j] = v[i * n + order[j]];
    }
  }
  return {std::move(values), std::move(vectors)};
}

} // namespace free_breakfast::geometry
