/// @file
/// @brief The host matrix products.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import :gemm;

namespace free_breakfast {

std::vector<double> gemm_tn(const std::vector<double> &a, const std::vector<double> &b, size_t k,
                            size_t m, size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      double sum = 0.0;
      for (size_t r = 0; r < k; ++r) {
        sum += a[r + (k * i)] * b[r + (k * j)];
      }
      c[i + (m * j)] = sum;
    }
  }
  return c;
}

std::vector<double> gemm_nn(const std::vector<double> &a, const std::vector<double> &b, size_t m,
                            size_t k, size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (size_t j = 0; j < n; ++j) {
    for (size_t r = 0; r < k; ++r) {
      const double w = b[r + (k * j)];
      for (size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += a[i + (m * r)] * w;
      }
    }
  }
  return c;
}

} // namespace free_breakfast
