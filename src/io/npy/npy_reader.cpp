/// @file
/// @brief Implementation of the .npy reader.
///
/// The format (https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html):
/// 6-byte magic "\x93NUMPY", a 2-byte version, a little-endian header length
/// (2 bytes in version 1, 4 in versions 2 and 3), a Python-dict header
/// carrying descr/fortran_order/shape, then the raw element bytes.

module free_breakfast.io.npy;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

// The payload is memcpy'd straight into doubles, which is only correct where
// '<f8' is the native representation.
static_assert(std::endian::native == std::endian::little,
              "the .npy module assumes a little-endian host");
static_assert(std::numeric_limits<double>::is_iec559,
              "the .npy module assumes IEEE-754 doubles");

namespace {

constexpr std::string_view kMagic{"\x93NUMPY", 6};

/// @brief A little-endian unsigned integer of @p nbytes bytes at @p offset.
[[nodiscard]] std::uint64_t read_le(std::string_view bytes, std::size_t offset,
                                    std::size_t nbytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < nbytes; ++i) {
    value |= std::uint64_t{static_cast<unsigned char>(bytes[offset + i])} << (8 * i);
  }
  return value;
}

/// @brief The header-dict value that follows @c 'key': with surrounding
/// spaces skipped, or npos-marked failure as an empty optional.
[[nodiscard]] std::optional<std::size_t> value_start(std::string_view header,
                                                     std::string_view key) {
  const auto key_pos = header.find(std::format("'{}'", key));
  if (key_pos == std::string_view::npos) {
    return std::nullopt;
  }
  auto pos = header.find(':', key_pos);
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  ++pos;
  while (pos < header.size() && header[pos] == ' ') {
    ++pos;
  }
  if (pos == header.size()) {
    return std::nullopt;
  }
  return pos;
}

/// @brief The quoted string value of @p key (e.g. descr's "<f8").
[[nodiscard]] Result<std::string_view> quoted_value(std::string_view header,
                                                    std::string_view key) {
  const auto start = value_start(header, key);
  if (!start || header[*start] != '\'') {
    return err_io("NPY header has no quoted '{}'", key);
  }
  const auto close = header.find('\'', *start + 1);
  if (close == std::string_view::npos) {
    return err_io("NPY header '{}' quote is unterminated", key);
  }
  return header.substr(*start + 1, close - *start - 1);
}

/// @brief The True/False value of @p key (fortran_order).
[[nodiscard]] Result<bool> bool_value(std::string_view header, std::string_view key) {
  const auto start = value_start(header, key);
  if (start && header.substr(*start).starts_with("True")) {
    return true;
  }
  if (start && header.substr(*start).starts_with("False")) {
    return false;
  }
  return err_io("NPY header has no boolean '{}'", key);
}

/// @brief The shape tuple, e.g. "()" -> {}, "(5,)" -> {5}, "(2, 3)" -> {2, 3}.
[[nodiscard]] Result<std::vector<std::size_t>> shape_value(std::string_view header) {
  const auto start = value_start(header, "shape");
  if (!start || header[*start] != '(') {
    return err_io("NPY header has no shape tuple");
  }
  const auto close = header.find(')', *start);
  if (close == std::string_view::npos) {
    return err_io("NPY header shape tuple is unterminated");
  }
  std::vector<std::size_t> shape;
  std::size_t pos = *start + 1;
  while (pos < close) {
    while (pos < close && (header[pos] == ' ' || header[pos] == ',')) {
      ++pos;
    }
    if (pos == close) {
      break;
    }
    std::size_t dim = 0;
    const char *first = header.data() + pos;
    const char *last = header.data() + close;
    const auto [ptr, ec] = std::from_chars(first, last, dim);
    if (ec != std::errc{} || ptr == first) {
      return err_io("NPY header shape entry is not a non-negative integer");
    }
    shape.push_back(dim);
    pos = static_cast<std::size_t>(ptr - header.data());
  }
  return shape;
}

} // namespace

[[nodiscard]] Result<NpyArray> parse_npy(std::string_view npy_bytes) {
  if (npy_bytes.size() < 8 || npy_bytes.substr(0, 6) != kMagic) {
    return err_io("not an NPY file: bad magic");
  }
  const unsigned major = static_cast<unsigned char>(npy_bytes[6]);
  if (major < 1 || major > 3) {
    return err_io("unsupported NPY format version {}", major);
  }
  const std::size_t len_bytes = (major == 1) ? 2 : 4;
  if (npy_bytes.size() < 8 + len_bytes) {
    return err_io("truncated NPY header length");
  }
  const std::uint64_t header_len = read_le(npy_bytes, 8, len_bytes);
  const std::size_t data_offset = 8 + len_bytes + header_len;
  if (npy_bytes.size() < data_offset) {
    return err_io("truncated NPY header");
  }
  const std::string_view header = npy_bytes.substr(8 + len_bytes, header_len);

  auto descr = FBF_TRY(quoted_value(header, "descr"));
  if (descr != "<f8") {
    return err_io("unsupported NPY dtype '{}' (only '<f8')", descr);
  }
  auto fortran = FBF_TRY(bool_value(header, "fortran_order"));
  if (fortran) {
    return err_io("Fortran-order NPY is not supported (C order only)");
  }
  auto shape = FBF_TRY(shape_value(header));

  auto count = FBF_TRY(element_count(shape));
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return err_config("NPY shape element count overflows");
  }
  const std::size_t expected = count * sizeof(double);
  const std::size_t actual = npy_bytes.size() - data_offset;
  if (actual != expected) {
    return err_io("NPY payload is {} bytes but the shape needs {}", actual, expected);
  }

  NpyArray array;
  array.shape = std::move(shape);
  array.data.resize(count);
  std::memcpy(array.data.data(), npy_bytes.data() + data_offset, expected);
  return array;
}

[[nodiscard]] Result<NpyArray> read_npy(const std::filesystem::path &path) {
  auto bytes = FBF_TRY(read_file(path));
  return parse_npy(bytes);
}

} // namespace free_breakfast
