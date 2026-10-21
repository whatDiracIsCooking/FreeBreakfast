/// @file
/// @brief The two host matrix products the SCF's l blocks need.

// An internal partition: nothing here is exported. The blocks are a handful of
// shells across, so plain loops, not BLAS.
module free_breakfast.experimental.atomic_scf:gemm;

import std;
import free_breakfast.core.types;

namespace free_breakfast {

/// @brief C = A^T B for A (k x m) and B (k x n), all column-major.
[[nodiscard]] std::vector<double>
gemm_tn(const std::vector<double> &a, const std::vector<double> &b, size_t k, size_t m, size_t n);

/// @brief C = A B for A (m x k) and B (k x n), all column-major.
[[nodiscard]] std::vector<double>
gemm_nn(const std::vector<double> &a, const std::vector<double> &b, size_t m, size_t k, size_t n);

} // namespace free_breakfast
