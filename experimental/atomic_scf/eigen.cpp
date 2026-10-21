/// @file
/// @brief Cyclic Jacobi for the l blocks.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import :eigen;

namespace free_breakfast {

SymmetricEigen jacobi_eigen(std::vector<double> a, size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    v[i + (n * i)] = 1.0;
  }
  for (int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < j; ++i) {
        off += a[i + (n * j)] * a[i + (n * j)];
      }
    }
    if (off < 1e-30) {
      break;
    }
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        const double apq = a[p + (n * q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double theta = (a[q + (n * q)] - a[p + (n * p)]) / (2.0 * apq);
        const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
        const double c = 1.0 / std::hypot(t, 1.0);
        const double s = t * c;
        for (size_t k = 0; k < n; ++k) {
          const double akp = a[k + (n * p)];
          const double akq = a[k + (n * q)];
          a[k + (n * p)] = (c * akp) - (s * akq);
          a[k + (n * q)] = (s * akp) + (c * akq);
        }
        for (size_t k = 0; k < n; ++k) {
          const double apk = a[p + (n * k)];
          const double aqk = a[q + (n * k)];
          a[p + (n * k)] = (c * apk) - (s * aqk);
          a[q + (n * k)] = (s * apk) + (c * aqk);
        }
        for (size_t k = 0; k < n; ++k) {
          const double vkp = v[k + (n * p)];
          const double vkq = v[k + (n * q)];
          v[k + (n * p)] = (c * vkp) - (s * vkq);
          v[k + (n * q)] = (s * vkp) + (c * vkq);
        }
      }
    }
  }
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::ranges::sort(order, [&](size_t x, size_t y) { return a[x + (n * x)] < a[y + (n * y)]; });
  SymmetricEigen out{std::vector<double>(n), std::vector<double>(n * n)};
  for (size_t j = 0; j < n; ++j) {
    out.values[j] = a[order[j] + (n * order[j])];
    std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(n * order[j]), n,
                out.vectors.begin() + static_cast<std::ptrdiff_t>(n * j));
  }
  return out;
}

} // namespace free_breakfast
