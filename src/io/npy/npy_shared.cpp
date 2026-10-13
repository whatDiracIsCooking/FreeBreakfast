/// @file
/// @brief Implementation of the shape arithmetic shared by both directions.

module free_breakfast.io.npy;

import std;
import free_breakfast.core.error_handling;

namespace free_breakfast {

[[nodiscard]] Result<std::size_t> element_count(std::span<const std::size_t> shape) {
  // The empty shape is a NumPy scalar: one element, not zero.
  std::size_t count = 1;
  for (const std::size_t dim : shape) {
    if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
      return err_config("NPY shape element count overflows");
    }
    count *= dim;
  }
  return count;
}

} // namespace free_breakfast
