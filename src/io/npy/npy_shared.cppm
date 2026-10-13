/// @file
/// @brief The array type shared by the .npy reader and writer.

export module free_breakfast.io.npy:shared;

import std;
import free_breakfast.core.error_handling;

export namespace free_breakfast {

/// @brief An n-dimensional float64 array in C (row-major) order.
///
/// The shape carries the dimensions exactly as the .npy header spells them:
/// `{}` is a NumPy scalar (one element), `{n}` a vector, `{rows, cols}` a
/// matrix. @c data holds the elements flattened in C order, so a matrix's
/// element (i, j) sits at `data[i * shape[1] + j]`.
struct NpyArray {
  std::vector<std::size_t> shape{};
  std::vector<double> data{};
};

/// @brief The number of elements a shape describes: the product of its
/// dimensions, with the empty shape (a NumPy scalar) counting one element.
///
/// Overflow of the product yields ErrorKind::InvalidConfig -- such a shape
/// cannot describe a real in-memory array.
[[nodiscard]] Result<std::size_t> element_count(std::span<const std::size_t> shape);

} // namespace free_breakfast
