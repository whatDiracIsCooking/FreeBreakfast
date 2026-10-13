/// @file
/// @brief Implementation of the shape arithmetic shared by both directions.

module free_breakfast.io.npy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

namespace free_breakfast {

[[nodiscard]] Result<size_t> element_count(std::span<const size_t> shape) {
  // The empty shape is a NumPy scalar: one element, not zero.
  size_t count = 1;
  for (const size_t dim : shape) {
    if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim) {
      return err_config("NPY shape element count overflows");
    }
    count *= dim;
  }
  return count;
}

} // namespace free_breakfast
