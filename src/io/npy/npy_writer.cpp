/// @file
/// @brief Implementation of the .npy writer.
///
/// Emits exactly what NumPy's np.save emits for a float64 Fortran-order array:
/// format version 1.0, the canonical header dict, and the header padded with
/// spaces (newline-terminated) so the data begins on a 64-byte boundary.

module free_breakfast.io.npy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import :format;

#include "diagnostics.h"

namespace free_breakfast {

static_assert(std::endian::native == std::endian::little,
              "the .npy module assumes a little-endian host");

/// @brief A shape as the header spells it: "()", "(5,)", "(2, 3)".
///
/// The one-dimensional trailing comma and the ", " separator match NumPy's
/// tuple repr byte for byte, which is what keeps npy_to_bytes identical to
/// np.save output.
[[nodiscard]] std::string format_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return "()";
  }
  if (shape.size() == 1) {
    return std::format("({},)", shape.front());
  }
  std::string out{"("};
  for (size_t i = 0; i < shape.size(); ++i) {
    out += std::format("{}{}", i == 0 ? "" : ", ", shape[i]);
  }
  out += ")";
  return out;
}

/// @brief Whether the C and Fortran layouts of @p shape differ: only when the
/// array is non-empty and at least two axes are longer than one.
///
/// np.save writes `'fortran_order': True` only then -- otherwise the array is
/// also C-contiguous, and NumPy checks that first -- so following the same
/// rule keeps npy_to_bytes identical to np.save for vectors, scalars and
/// degenerate matrices too.
[[nodiscard]] bool layouts_differ(std::span<const size_t> shape) {
  if (std::ranges::contains(shape, size_t{0})) {
    return false;
  }
  return std::ranges::count_if(shape, [](size_t dim) { return dim > 1; }) > 1;
}

[[nodiscard]] Result<std::string> npy_to_bytes(const NpyArray &array) {
  auto count = FBF_TRY(element_count(array.shape));
  if (count != array.data.size()) {
    return err_config("NPY shape {} describes {} elements but data has {}",
                      format_shape(array.shape), count, array.data.size());
  }

  // The canonical dict, then space padding so magic + version + length field
  // + header is a multiple of 64, with the final byte a newline.
  const std::string dict =
      std::format("{{'descr': '<f8', 'fortran_order': {}, 'shape': {}, }}",
                  layouts_differ(array.shape) ? "True" : "False", format_shape(array.shape));
  const size_t unpadded = 6 + 2 + 2 + dict.size() + 1;
  const size_t padded = ((unpadded + 63) / 64) * 64;
  const size_t header_len = padded - (6 + 2 + 2);
  if (header_len > std::numeric_limits<uint16_t>::max()) {
    return err_config("NPY header does not fit format version 1.0");
  }

  std::string out;
  out.reserve(padded + array.data.size() * sizeof(double));
  out.append(kMagic);
  out.push_back('\x01'); // format version 1.0
  out.push_back('\x00');
  out.push_back(static_cast<char>(header_len & 0xff)); // little-endian uint16
  out.push_back(static_cast<char>((header_len >> 8) & 0xff));
  out += dict;
  out.append(header_len - dict.size() - 1, ' ');
  out.push_back('\n');
  out.append(reinterpret_cast<const char *>(array.data.data()), array.data.size() * sizeof(double));
  return out;
}

[[nodiscard]] Status write_npy(const std::filesystem::path &path, const NpyArray &array) {
  auto bytes = FBF_TRY(npy_to_bytes(array));
  return write_file(path, bytes);
}

} // namespace free_breakfast
