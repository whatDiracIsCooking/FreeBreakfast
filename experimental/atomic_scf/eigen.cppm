/// @file
/// @brief A symmetric eigensolver for the SCF's l blocks.

// An internal partition: nothing here is exported.
module free_breakfast.experimental.atomic_scf:eigen;

import std;
import free_breakfast.core.types;

namespace free_breakfast {

/// @brief The eigendecomposition of a symmetric n x n matrix.
struct SymmetricEigen {
  std::vector<double> values;  ///< Ascending.
  std::vector<double> vectors; ///< n x n, column-major, column j for values[j].
};

/// @brief The eigendecomposition of the symmetric @p n x @p n column-major
/// @p a, by cyclic Jacobi.
///
/// The l blocks are a handful of shells, where Jacobi is exact to round-off
/// and needs no LAPACK.
[[nodiscard]] SymmetricEigen jacobi_eigen(std::vector<double> a, size_t n);

} // namespace free_breakfast
