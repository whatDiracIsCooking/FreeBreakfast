/// @file
/// @brief The two host matrix products the SCF's l blocks need.

// An internal partition of free_breakfast.sad. The blocks are a
// handful of shells across, so plain loops, not BLAS.
module free_breakfast.sad:gemm;

import std;
import free_breakfast.core.types;

namespace free_breakfast {

/// @brief C = A^T B for A (k x m) and B (k x n), all column-major.
[[nodiscard]] std::vector<double>
gemm_tn(const std::vector<double> &a, const std::vector<double> &b, size_t k, size_t m, size_t n) {
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

/// @brief C = A B for A (m x k) and B (k x n), all column-major.
[[nodiscard]] std::vector<double>
gemm_nn(const std::vector<double> &a, const std::vector<double> &b, size_t m, size_t k, size_t n) {
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
